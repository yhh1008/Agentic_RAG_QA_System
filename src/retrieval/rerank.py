from __future__ import annotations

from dataclasses import dataclass

from src.config.settings import settings


@dataclass
class RetrievedItem:
    doc_id: str
    chunk_id: str
    source: str
    start_offset: int
    end_offset: int
    content: str
    distance: float
    hit_keyword: bool = False


def apply_keyword_bonus(items: list[RetrievedItem], keyword: str) -> list[RetrievedItem]:
    if not keyword:
        return sorted(items, key=lambda x: x.distance)

    lowered_keyword = keyword.lower().strip()
    for item in items:
        if lowered_keyword and lowered_keyword in item.content.lower():
            item.hit_keyword = True
            item.distance -= settings.keyword_bonus

    return sorted(items, key=lambda x: x.distance)
