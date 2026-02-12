from __future__ import annotations

"""
SFT 训练入口骨架（LoRA/QLoRA）。
实际训练可接 TRL + PEFT，这里先提供统一参数入口与数据读取格式。
"""

import json
from pathlib import Path


DATA_FILE = Path("train/data/sft_trajectories.jsonl")


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"missing {DATA_FILE}")

    samples = [json.loads(x) for x in DATA_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]
    print(f"loaded_samples={len(samples)}")
    print("TODO: integrate TRL SFTTrainer + PEFT LoRA here")


if __name__ == "__main__":
    main()
