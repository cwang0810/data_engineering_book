from __future__ import annotations

from collections import Counter, defaultdict

from pipeline_utils import (
    PROCESSED_DIR,
    TRAINING_DIR,
    deterministic_bucket,
    estimated_tokens,
    ensure_dir,
    load_jsonl,
    write_json,
    write_jsonl,
)

INPUT_FILES = [
    PROCESSED_DIR / "llava_instruct.jsonl",
    PROCESSED_DIR / "llava_alignment.jsonl",
    PROCESSED_DIR / "llava_interleaved.jsonl",
]
LOW_QUALITY_FILE = PROCESSED_DIR / "low_quality_flags.jsonl"
FINAL_FILE = TRAINING_DIR / "final_llava_dataset.jsonl"
TRAIN_FILE = TRAINING_DIR / "train.jsonl"
VAL_FILE = TRAINING_DIR / "val.jsonl"
SMOKE_FILE = TRAINING_DIR / "smoke_test.jsonl"
MANIFEST_FILE = TRAINING_DIR / "training_manifest.json"


def main() -> None:
    ensure_dir(TRAINING_DIR)
    blocked_ids = {record["id"] for record in load_jsonl(LOW_QUALITY_FILE)}

    merged: list[dict] = []
    for path in INPUT_FILES:
        merged.extend(load_jsonl(path))

    final_records = [record for record in merged if record["id"] not in blocked_ids]
    final_records = sorted(final_records, key=lambda item: item["id"])

    train_records: list[dict] = []
    val_records: list[dict] = []
    smoke_by_task: dict[str, list[dict]] = defaultdict(list)

    for record in final_records:
        if deterministic_bucket(record["id"]) < 90:
            train_records.append(record)
        else:
            val_records.append(record)
        if len(smoke_by_task[record["task_type"]]) < 3:
            smoke_by_task[record["task_type"]].append(record)

    smoke_records: list[dict] = []
    for task_type in sorted(smoke_by_task):
        smoke_records.extend(smoke_by_task[task_type])
    smoke_records = smoke_records[:24]

    total_tokens = sum(
        estimated_tokens(" ".join(turn["value"] for turn in record["conversations"]))
        for record in final_records
    )

    manifest = {
        "num_records": len(final_records),
        "num_train_records": len(train_records),
        "num_val_records": len(val_records),
        "num_smoke_records": len(smoke_records),
        "task_distribution": dict(Counter(record["task_type"] for record in final_records)),
        "asset_type_distribution": dict(Counter(record["asset_type"] for record in final_records)),
        "estimated_tokens_total": total_tokens,
    }

    write_jsonl(final_records, FINAL_FILE)
    write_jsonl(train_records, TRAIN_FILE)
    write_jsonl(val_records, VAL_FILE)
    write_jsonl(smoke_records, SMOKE_FILE)
    write_json(manifest, MANIFEST_FILE)
    print("✅ 训练前准备完成。")
    print(manifest)


if __name__ == "__main__":
    main()
