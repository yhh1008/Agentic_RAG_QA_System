from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    trace_id: str
    query: str
    session_id: str | None
    is_policy_related: bool

    expand_query: str
    keyword: str

    retrieved_items: list[dict[str, Any]]
    evidence: list[dict[str, Any]]

    attempts: int
    used_tools: list[str]

    answer: str
    citations: list[dict[str, Any]]
    is_answerable: bool
