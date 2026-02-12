from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from pydantic import BaseModel, Field

from langgraph.graph import END, StateGraph

from src.agent.prompts import ANSWER_PROMPT, CLASSIFY_PROMPT, FALLBACK_ANSWER, SYSTEM_PROMPT
from src.agent.state import AgentState
from src.agent.tools import (
    EvidenceItem,
    ExpandKeywordOutput,
    RetrieveItemOutput,
    SelectiveReadOutput,
    expand_and_keyword,
    retrieval_augment,
    summary_related_doc,
)
from src.config.settings import settings
from src.llm.deepseek_client import get_chat_model, invoke_structured


def _append_trace(trace_id: str, event: dict) -> None:
    path = Path(settings.trace_dir) / f"{trace_id}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _log_tool(state: AgentState, tool_name: str, args: dict, output: dict) -> None:
    _append_trace(
        state["trace_id"],
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": "tool_call",
            "tool": tool_name,
            "args": args,
            "output": output,
            "attempt": state.get("attempts", 1),
        },
    )


def classify_query_node(state: AgentState) -> AgentState:
    class ClassifyOutput(BaseModel):
        is_policy_related: bool = Field(...)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": CLASSIFY_PROMPT.format(query=state["query"])},
    ]
    res = invoke_structured(messages, ClassifyOutput)
    return {"is_policy_related": res.is_policy_related}


def rewrite_keyword_node(state: AgentState) -> AgentState:
    res: ExpandKeywordOutput = expand_and_keyword(state["query"])
    output = {"expand_query": res.expand_query, "keyword": res.keyword}
    _log_tool(state, "expand_and_keyword", {"query": state["query"]}, output)
    return {
        "expand_query": res.expand_query,
        "keyword": res.keyword,
        "used_tools": state.get("used_tools", []) + ["expand_and_keyword"],
    }


def retrieve_node(state: AgentState) -> AgentState:
    query = state.get("expand_query") or state["query"]
    keyword = state.get("keyword", "")
    res = retrieval_augment(query, keyword)
    output = {"items": [x.model_dump() for x in res.items]}
    _log_tool(state, "retrieval_augment", {"query": query, "keyword": keyword}, output)
    return {
        "retrieved_items": output["items"],
        "used_tools": state.get("used_tools", []) + ["retrieval_augment"],
    }


def selective_read_node(state: AgentState) -> AgentState:
    items = [RetrieveItemOutput(**x) for x in state.get("retrieved_items", [])]
    total_chars = sum(len(x.content) for x in items)
    if total_chars <= settings.selective_read_char_threshold:
        evidence = [
            EvidenceItem(
                doc_id=x.doc_id,
                chunk_id=x.chunk_id,
                source=x.source,
                excerpt=x.content,
            ).model_dump()
            for x in items
        ]
        return {"evidence": evidence}

    res: SelectiveReadOutput = summary_related_doc(state["query"], items)
    output = {"evidence": [x.model_dump() for x in res.evidence]}
    _log_tool(
        state,
        "summary_related_doc",
        {"query": state["query"], "related_count": len(items)},
        output,
    )
    return {
        "evidence": output["evidence"],
        "used_tools": state.get("used_tools", []) + ["summary_related_doc"],
    }


def answer_node(state: AgentState) -> AgentState:
    class Citation(BaseModel):
        doc_id: str
        chunk_id: str
        source: str
        quote: str

    class AgentAnswerOutput(BaseModel):
        answer: str
        citations: list[Citation]
        is_answerable: bool

    if not state.get("is_policy_related", False):
        model = get_chat_model()
        content = model.invoke(
            [
                {"role": "system", "content": "你是一个中文问答助手。"},
                {"role": "user", "content": state["query"]},
            ]
        ).content
        return {
            "answer": content,
            "citations": [],
            "is_answerable": True,
        }

    evidence = state.get("evidence", [])
    if not evidence:
        return {"answer": FALLBACK_ANSWER, "citations": [], "is_answerable": False}

    messages = [
        {"role": "system", "content": ANSWER_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "query": state["query"],
                    "evidence": evidence,
                    "requirements": "严格引用证据，中文回答",
                },
                ensure_ascii=False,
            ),
        },
    ]
    res = invoke_structured(messages, AgentAnswerOutput)
    return {
        "answer": res.answer,
        "citations": [x.model_dump() for x in res.citations],
        "is_answerable": res.is_answerable,
    }


def retry_or_end(state: AgentState) -> str:
    if state.get("is_answerable", False):
        return "end"
    if state.get("attempts", 1) >= settings.max_attempts:
        return "end"
    return "retry"


def increment_attempt_node(state: AgentState) -> AgentState:
    return {"attempts": state.get("attempts", 1) + 1}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_query_node)
    graph.add_node("rewrite", rewrite_keyword_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("selective_read", selective_read_node)
    graph.add_node("answer", answer_node)
    graph.add_node("retry_inc", increment_attempt_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "selective_read")
    graph.add_edge("selective_read", "answer")

    graph.add_conditional_edges(
        "answer",
        retry_or_end,
        {
            "retry": "retry_inc",
            "end": END,
        },
    )
    graph.add_edge("retry_inc", "rewrite")

    return graph.compile()


def run_agent(query: str, session_id: str | None = None) -> dict:
    trace_id = str(uuid.uuid4())
    app = build_graph()
    init_state: AgentState = {
        "trace_id": trace_id,
        "query": query,
        "session_id": session_id,
        "attempts": 1,
        "used_tools": [],
        "retrieved_items": [],
        "evidence": [],
        "citations": [],
    }
    out = app.invoke(init_state)
    return {
        "answer": out.get("answer", FALLBACK_ANSWER),
        "citations": out.get("citations", []),
        "trace_id": trace_id,
        "attempts": out.get("attempts", 1),
        "used_tools": out.get("used_tools", []),
    }
