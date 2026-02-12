from __future__ import annotations

import json
from pathlib import Path

from src.agent.graph import run_agent


INPUT_FILE = Path("train/data/sample_eval.jsonl")
OUTPUT_FILE = Path("train/data/sft_trajectories.jsonl")


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in INPUT_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for row in rows:
            result = run_agent(row["query"])
            sample = {
                "query": row["query"],
                "trajectory_ref": result["trace_id"],
                "answer": result["answer"],
                "citations": result["citations"],
                "used_tools": result["used_tools"],
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"saved={OUTPUT_FILE}")


if __name__ == "__main__":
    main()
