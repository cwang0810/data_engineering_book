from __future__ import annotations

import json
import os
from pathlib import Path

from pipeline_utils import estimate_token_count, normalize_text, sha1_text

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TRAINING_DIR = DATA_DIR / "training"

INPUT_FILE = PROCESSED_DIR / "final_data.jsonl"
SERIALIZED_FILE = TRAINING_DIR / "serialized_dataset.jsonl"
TRAIN_FILE = TRAINING_DIR / "train.jsonl"
VAL_FILE = TRAINING_DIR / "val.jsonl"
SMOKE_TEST_FILE = TRAINING_DIR / "smoke_test.jsonl"
MANIFEST_FILE = TRAINING_DIR / "training_manifest.json"

TRAIN_DIR = TRAINING_DIR / "train_shards"
VAL_DIR = TRAINING_DIR / "val_shards"

VAL_RATIO_PERCENT = 10
SHARD_SIZE = 256
SMOKE_TEST_SIZE = 32


def assign_split(text_sha1: str) -> str:
    bucket = int(text_sha1[:8], 16) % 100
    return "val" if bucket < VAL_RATIO_PERCENT else "train"


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_shards(records: list[dict], output_dir: Path, prefix: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for shard_idx, start in enumerate(range(0, len(records), SHARD_SIZE)):
        shard_records = records[start:start + SHARD_SIZE]
        shard_path = output_dir / f"{prefix}-{shard_idx:05d}.jsonl"
        write_jsonl(shard_path, shard_records)
        written.append(str(shard_path.relative_to(PROJECT_ROOT)))
    return written


def build_records() -> list[dict]:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    records = []
    seen_hashes = set()
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = normalize_text(item.get("text", ""))
            if not text:
                continue

            text_sha1 = sha1_text(text)
            if text_sha1 in seen_hashes:
                continue
            seen_hashes.add(text_sha1)

            record = {
                "id": f"mini_c4_{len(records):06d}",
                "source_line": line_number,
                "url": item.get("url"),
                "lang": item.get("lang", "unknown"),
                "text": text,
                "text_sha1": text_sha1,
                "n_chars": len(text),
                "estimated_tokens": estimate_token_count(text),
            }
            record["split"] = assign_split(text_sha1)
            records.append(record)

    return records


def validate_records(train_records: list[dict], val_records: list[dict]) -> dict:
    train_hashes = {record["text_sha1"] for record in train_records}
    val_hashes = {record["text_sha1"] for record in val_records}
    overlap = train_hashes & val_hashes
    if overlap:
        raise ValueError(f"Train/val overlap detected: {len(overlap)} items")

    if any(not record["text"].strip() for record in train_records + val_records):
        raise ValueError("Empty text detected after serialization")

    return {
        "train_hashes": len(train_hashes),
        "val_hashes": len(val_hashes),
        "overlap": len(overlap),
    }


def build_smoke_test_records(train_records: list[dict]) -> list[dict]:
    by_lang = {}
    for record in train_records:
        by_lang.setdefault(record["lang"], []).append(record)

    smoke_test = []
    lang_keys = sorted(by_lang.keys())
    index_by_lang = {lang: 0 for lang in lang_keys}

    while len(smoke_test) < min(SMOKE_TEST_SIZE, len(train_records)):
        made_progress = False
        for lang in lang_keys:
            current_index = index_by_lang[lang]
            if current_index >= len(by_lang[lang]):
                continue
            smoke_test.append(by_lang[lang][current_index])
            index_by_lang[lang] += 1
            made_progress = True
            if len(smoke_test) >= SMOKE_TEST_SIZE:
                break
        if not made_progress:
            break

    return smoke_test


def main() -> None:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    records = build_records()
    train_records = [record for record in records if record["split"] == "train"]
    val_records = [record for record in records if record["split"] == "val"]
    smoke_test_records = build_smoke_test_records(train_records)

    validation = validate_records(train_records, val_records)

    write_jsonl(SERIALIZED_FILE, records)
    write_jsonl(TRAIN_FILE, train_records)
    write_jsonl(VAL_FILE, val_records)
    write_jsonl(SMOKE_TEST_FILE, smoke_test_records)

    train_shards = write_shards(train_records, TRAIN_DIR, "train")
    val_shards = write_shards(val_records, VAL_DIR, "val")

    lang_distribution = {}
    for record in records:
        lang_distribution[record["lang"]] = lang_distribution.get(record["lang"], 0) + 1

    manifest = {
        "input_file": str(INPUT_FILE.relative_to(PROJECT_ROOT)),
        "serialized_file": str(SERIALIZED_FILE.relative_to(PROJECT_ROOT)),
        "train_file": str(TRAIN_FILE.relative_to(PROJECT_ROOT)),
        "val_file": str(VAL_FILE.relative_to(PROJECT_ROOT)),
        "smoke_test_file": str(SMOKE_TEST_FILE.relative_to(PROJECT_ROOT)),
        "num_records": len(records),
        "num_train_records": len(train_records),
        "num_val_records": len(val_records),
        "num_smoke_test_records": len(smoke_test_records),
        "estimated_tokens_total": sum(record["estimated_tokens"] for record in records),
        "estimated_tokens_train": sum(record["estimated_tokens"] for record in train_records),
        "estimated_tokens_val": sum(record["estimated_tokens"] for record in val_records),
        "lang_distribution": lang_distribution,
        "validation": validation,
        "train_shards": train_shards,
        "val_shards": val_shards,
    }

    with MANIFEST_FILE.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Training data preparation complete.")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
