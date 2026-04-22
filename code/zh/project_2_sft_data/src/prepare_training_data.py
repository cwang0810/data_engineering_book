from __future__ import annotations

import json

from pipeline_utils import deterministic_bucket, estimated_tokens, load_jsonl, processed_dir, training_dir, write_jsonl


PROCESSED_DIR = processed_dir()
TRAINING_DIR = training_dir()

ACCEPTED_FILE = PROCESSED_DIR / "domain_expert_sft.jsonl"
RISK_REFUSAL_FILE = PROCESSED_DIR / "legal_risk_refusal_sft.jsonl"

FINAL_DATASET_FILE = TRAINING_DIR / "final_sft_dataset.jsonl"
TRAIN_FILE = TRAINING_DIR / "train.jsonl"
VAL_FILE = TRAINING_DIR / "val.jsonl"
SMOKE_FILE = TRAINING_DIR / "smoke_test.jsonl"
MANIFEST_FILE = TRAINING_DIR / "training_manifest.json"

VAL_PERCENT = 10
SMOKE_SIZE = 24


def main() -> None:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    accepted = load_jsonl(ACCEPTED_FILE)
    risk_refusals = load_jsonl(RISK_REFUSAL_FILE)
    all_records = accepted + risk_refusals

    final_records = []
    for idx, item in enumerate(all_records):
        record = {
            "id": f"legal_sft_{idx:06d}",
            "instruction": item["instruction"],
            "input": item.get("input", ""),
            "output": item["output"],
            "task_type": item["task_type"],
            "domain": item.get("domain", "legal"),
            "law_name": item.get("law_name", "unknown"),
            "article_no": item.get("article_no", "N/A"),
            "source_doc": item.get("source_doc", "unknown"),
        }
        split = "val" if deterministic_bucket(record["instruction"] + record["output"]) < VAL_PERCENT else "train"
        record["split"] = split
        record["estimated_tokens"] = estimated_tokens(record["instruction"] + "\n" + record["output"])
        final_records.append(record)

    train_records = [record for record in final_records if record["split"] == "train"]
    val_records = [record for record in final_records if record["split"] == "val"]

    smoke_records = []
    by_task = {}
    for record in train_records:
        by_task.setdefault(record["task_type"], []).append(record)
    while len(smoke_records) < min(SMOKE_SIZE, len(train_records)):
        progressed = False
        for task_type in sorted(by_task.keys()):
            if not by_task[task_type]:
                continue
            smoke_records.append(by_task[task_type].pop(0))
            progressed = True
            if len(smoke_records) >= SMOKE_SIZE:
                break
        if not progressed:
            break

    write_jsonl(FINAL_DATASET_FILE, final_records)
    write_jsonl(TRAIN_FILE, train_records)
    write_jsonl(VAL_FILE, val_records)
    write_jsonl(SMOKE_FILE, smoke_records)

    manifest = {
        "num_records": len(final_records),
        "num_train_records": len(train_records),
        "num_val_records": len(val_records),
        "num_smoke_records": len(smoke_records),
        "task_distribution": {},
        "estimated_tokens_total": sum(record["estimated_tokens"] for record in final_records),
    }
    for record in final_records:
        manifest["task_distribution"][record["task_type"]] = manifest["task_distribution"].get(record["task_type"], 0) + 1

    with MANIFEST_FILE.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("✅ 训练前准备完成。")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
