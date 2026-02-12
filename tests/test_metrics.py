from src.eval.metrics import citation_validity, retrieval_hit_at_k, tool_call_accuracy


def test_tool_call_accuracy():
    rows = [{"tool_correct": True}, {"tool_correct": False}, {"tool_correct": True}]
    assert abs(tool_call_accuracy(rows) - 2 / 3) < 1e-6


def test_retrieval_hit_at_k():
    rows = [
        {"predicted_chunks": ["a", "b"], "gold_chunks": ["b"]},
        {"predicted_chunks": ["x"], "gold_chunks": ["y"]},
    ]
    assert abs(retrieval_hit_at_k(rows, k=2) - 0.5) < 1e-6


def test_citation_validity():
    rows = [{"citation_valid": True}, {"citation_valid": False}]
    assert abs(citation_validity(rows) - 0.5) < 1e-6
