from __future__ import annotations

import json
import random
from pathlib import Path


DEFAULT_FILE = Path(__file__).resolve().parent.parent / "data" / "processed" / "domain_expert_sft.jsonl"


def view_random_samples(path: Path, n: int = 5) -> None:
    if not path.exists():
        print(f"错误：找不到文件 {path}")
        return

    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        print("文件是空的。")
        return

    samples = random.sample(lines, min(n, len(lines)))
    print(f"--- 随机抽检 {len(samples)} 条数据: {path.name} ---")
    for i, line in enumerate(samples, start=1):
        data = json.loads(line)
        print(f"\n[样本 {i}]")
        print(f"任务: {data.get('task_type')}")
        print(f"法律: {data.get('law_name')}")
        print(f"指令: {data.get('instruction', '')[:120]}...")
        print(f"回答: {data.get('output', '')[:180]}...")


if __name__ == "__main__":
    view_random_samples(DEFAULT_FILE)
