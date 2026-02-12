from __future__ import annotations

from pydantic import BaseModel, Field

from src.config.settings import settings
from src.llm.deepseek_client import invoke_structured
from src.retrieval.chroma_store import get_vectorstore
from src.retrieval.rerank import RetrievedItem, apply_keyword_bonus
from src.agent.prompts import EXPAND_KEYWORD_PROMPT, SELECTIVE_READ_PROMPT


class ExpandKeywordOutput(BaseModel):
    expand_query: str = Field(..., description="对query进行扩写/改写的结果")
    keyword: str = Field(..., description="最重要的简短关键词")


class RetrieveItemOutput(BaseModel):
    doc_id: str
    chunk_id: str
    source: str
    start_offset: int
    end_offset: int
    content: str
    distance: float
    hit_keyword: bool


class RetrieveOutput(BaseModel):
    items: list[RetrieveItemOutput]


class EvidenceItem(BaseModel):
    doc_id: str
    chunk_id: str
    source: str
    excerpt: str


class SelectiveReadOutput(BaseModel):
    evidence: list[EvidenceItem]


def expand_and_keyword(query: str) -> ExpandKeywordOutput:
    messages = [
        {"role": "system", "content": "你负责改写query并提取关键词。"},
        {"role": "user", "content": EXPAND_KEYWORD_PROMPT.format(query=query)},
    ]
    return invoke_structured(messages, ExpandKeywordOutput)


def retrieval_augment(query: str, keyword: str) -> RetrieveOutput:
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search_with_score(query, k=settings.top_k_recall)

    items: list[RetrievedItem] = []
    for doc, distance in docs:
        metadata = doc.metadata or {}
        items.append(
            RetrievedItem(
                doc_id=str(metadata.get("doc_id", "unknown_doc")),
                chunk_id=str(metadata.get("chunk_id", "unknown_chunk")),
                source=str(metadata.get("source", "unknown_source")),
                start_offset=int(metadata.get("start_offset", 0)),
                end_offset=int(metadata.get("end_offset", 0)),
                content=doc.page_content,
                distance=float(distance),
            )
        )

    ranked = apply_keyword_bonus(items, keyword)[: settings.top_k_final]
    return RetrieveOutput(
        items=[RetrieveItemOutput(**item.__dict__) for item in ranked]
    )


def summary_related_doc(query: str, related_items: list[RetrieveItemOutput]) -> SelectiveReadOutput:
    related_doc = "\n\n".join(
        [f"[{i.doc_id}:{i.chunk_id}] {i.content}" for i in related_items]
    )

    class _SummarySchema(BaseModel):
        evidence: list[EvidenceItem]

    messages = [
        {"role": "system", "content": SELECTIVE_READ_PROMPT},
        {
            "role": "user",
            "content": f"用户问题：{query}\n\n原文片段：\n{related_doc}",
        },
    ]
    return invoke_structured(messages, _SummarySchema)
