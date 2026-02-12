from __future__ import annotations


def tool_call_accuracy(records: list[dict]) -> float:
    if not records:
        return 0.0
    correct = sum(1 for r in records if r.get("tool_correct", False))
    return correct / len(records)


def retrieval_hit_at_k(records: list[dict], k: int = 5) -> float:
    if not records:
        return 0.0
    hit = 0
    for r in records:
        predicted = set(r.get("predicted_chunks", [])[:k])
        gold = set(r.get("gold_chunks", []))
        if predicted.intersection(gold):
            hit += 1
    return hit / len(records)


def citation_validity(records: list[dict]) -> float:
    if not records:
        return 0.0
    valid = sum(1 for r in records if r.get("citation_valid", False))
    return valid / len(records)


def answerable_accuracy(records: list[dict]) -> float:
    if not records:
        return 0.0
    correct = sum(1 for r in records if r.get("answerable_correct", False))
    return correct / len(records)
