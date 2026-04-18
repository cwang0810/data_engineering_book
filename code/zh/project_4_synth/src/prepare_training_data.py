from __future__ import annotations

from collections import Counter, defaultdict

from pipeline_utils import (
    PROCESSED_DIR,
    TRAINING_DIR,
    deterministic_bucket,
    estimated_tokens,
    ensure_standard_dirs,
    load_jsonl,
    write_json,
    write_jsonl,
)

INPUT_FILE = PROCESSED_DIR / "verified_textbook.jsonl"
LOW_QUALITY_FILE = PROCESSED_DIR / "low_quality_flags.jsonl"
FINAL_FILE = TRAINING_DIR / "final_textbook_dataset.jsonl"
TRAIN_FILE = TRAINING_DIR / "train.jsonl"
VAL_FILE = TRAINING_DIR / "val.jsonl"
SMOKE_FILE = TRAINING_DIR / "smoke_test.jsonl"
MANIFEST_FILE = TRAINING_DIR / "training_manifest.json"


def make_training_record(record: dict) -> dict:
    instruction = (
        f"Write a {record['domain']} textbook section for the topic `{record['topic']}` "
        f"at `{record['difficulty']}` difficulty. Include explanation, exercise, and verified answer."
    )
    response = record["chapter_markdown"]
    return {
        "id": record["id"],
        "domain": record["domain"],
        "topic": record["topic"],
        "difficulty": record["difficulty"],
        "instruction": instruction,
        "response": response,
        "source_dataset": record["source_dataset"],
    }


def main() -> None:
    ensure_standard_dirs()
    blocked = {record["id"] for record in load_jsonl(LOW_QUALITY_FILE)}
    verified = [record for record in load_jsonl(INPUT_FILE) if record["id"] not in blocked]
    final_records = [make_training_record(record) for record in verified]
    final_records = sorted(final_records, key=lambda item: item["id"])

    train_records: list[dict] = []
    val_records: list[dict] = []
    smoke_by_domain: dict[str, list[dict]] = defaultdict(list)

    for record in final_records:
        if deterministic_bucket(record["id"]) < 85:
            train_records.append(record)
        else:
            val_records.append(record)
        if len(smoke_by_domain[record["domain"]]) < 6:
            smoke_by_domain[record["domain"]].append(record)

    smoke_records: list[dict] = []
    for domain in sorted(smoke_by_domain):
        smoke_records.extend(smoke_by_domain[domain])

    total_tokens = sum(
        estimated_tokens(record["instruction"] + "\n" + record["response"]) for record in final_records
    )

    manifest = {
        "num_records": len(final_records),
        "num_train_records": len(train_records),
        "num_val_records": len(val_records),
        "num_smoke_records": len(smoke_records),
        "domain_distribution": dict(Counter(record["domain"] for record in final_records)),
        "topic_distribution": dict(Counter(record["topic"] for record in final_records)),
        "difficulty_distribution": dict(Counter(record["difficulty"] for record in final_records)),
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
