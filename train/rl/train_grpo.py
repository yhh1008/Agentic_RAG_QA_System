from __future__ import annotations

"""
GRPO/RL 训练入口骨架。
这里预留奖励函数接口与轨迹读取结构，后续接 VeRL。
"""

import json
from pathlib import Path


DATA_FILE = Path("train/data/sft_trajectories.jsonl")


def reward_fn(sample: dict) -> float:
    tool_reward = 1.0 if sample.get("used_tools") else 0.0
    citation_reward = 1.0 if sample.get("citations") else 0.0
    return 0.6 * tool_reward + 0.4 * citation_reward


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"missing {DATA_FILE}")

    samples = [json.loads(x) for x in DATA_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]
    rewards = [reward_fn(s) for s in samples]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"loaded_samples={len(samples)} avg_reward={avg_reward:.4f}")
    print("TODO: integrate VeRL GRPO trainer here")


if __name__ == "__main__":
    main()
