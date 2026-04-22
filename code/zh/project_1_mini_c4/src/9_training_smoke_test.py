from __future__ import annotations

import json
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SMOKE_TEST_FILE = PROJECT_ROOT / "data" / "training" / "smoke_test.jsonl"


def main() -> None:
    if not SMOKE_TEST_FILE.exists():
        raise FileNotFoundError(
            f"Smoke test file not found: {SMOKE_TEST_FILE}. Run 7_prepare_training_data.py first."
        )

    records = []
    ids = set()
    with SMOKE_TEST_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            records.append(item)
            if item["id"] in ids:
                raise ValueError(f"Duplicate id detected in smoke test: {item['id']}")
            ids.add(item["id"])
            if not item.get("text", "").strip():
                raise ValueError(f"Empty text detected in smoke test record: {item['id']}")
            if item.get("estimated_tokens", 0) <= 0:
                raise ValueError(f"Non-positive token estimate in smoke test record: {item['id']}")

    summary = {
        "num_records": len(records),
        "languages": sorted({record.get("lang", "unknown") for record in records}),
        "min_tokens": min(record["estimated_tokens"] for record in records) if records else 0,
        "max_tokens": max(record["estimated_tokens"] for record in records) if records else 0,
    }

    print("Training smoke test passed.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
