from __future__ import annotations

from collections import Counter, defaultdict

from pipeline_utils import PROCESSED_DIR, TRAINING_DIR, deterministic_bucket, ensure_standard_dirs, estimated_tokens, load_jsonl, write_json, write_jsonl

INPUT_FILE = PROCESSED_DIR / "validated_traces.jsonl"
STEP_REWARD_FILE = PROCESSED_DIR / "step_rewards.jsonl"
FINAL_FILE = TRAINING_DIR / "prm_step_dataset.jsonl"
TRAIN_FILE = TRAINING_DIR / "train.jsonl"
VAL_FILE = TRAINING_DIR / "val.jsonl"
SMOKE_FILE = TRAINING_DIR / "smoke_test.jsonl"
MANIFEST_FILE = TRAINING_DIR / "training_manifest.json"


def build_step_record(trace: dict, step: dict) -> dict:
    prompt = trace["question"]
    previous_steps = [item["text"] for item in trace["steps"] if item["step_idx"] < step["step_idx"]]
    return {
        "record_id": f"{trace['trace_id']}_step_{step['step_idx']}",
        "trace_id": trace["trace_id"],
        "seed_id": trace["seed_id"],
        "domain": trace["domain"],
        "topic": trace["topic"],
        "trace_type": trace["trace_type"],
        "prompt": prompt,
        "previous_steps": previous_steps,
        "current_step": step["text"],
        "label": step["label"],
        "step_kind": step["kind"],
        "trace_score": trace["trace_score"],
        "reward_bucket": trace["reward_bucket"],
        "validation_passed": trace["validation_passed"],
    }


def main() -> None:
    ensure_standard_dirs()
    traces = load_jsonl(INPUT_FILE)
    step_records: list[dict] = []
    for trace in traces:
        for step in trace["steps"]:
            step_records.append(build_step_record(trace, step))

    step_records = sorted(step_records, key=lambda item: item["record_id"])
    train_records: list[dict] = []
    val_records: list[dict] = []
    smoke_by_domain: dict[str, list[dict]] = defaultdict(list)

    for record in step_records:
        if deterministic_bucket(record["trace_id"]) < 85:
            train_records.append(record)
        else:
            val_records.append(record)
        if len(smoke_by_domain[record["domain"]]) < 8:
            smoke_by_domain[record["domain"]].append(record)

    smoke_records: list[dict] = []
    for domain in sorted(smoke_by_domain):
        smoke_records.extend(smoke_by_domain[domain])

    manifest = {
        "num_records": len(step_records),
        "num_train_records": len(train_records),
        "num_val_records": len(val_records),
        "num_smoke_records": len(smoke_records),
        "domain_distribution": dict(Counter(record["domain"] for record in step_records)),
        "trace_type_distribution": dict(Counter(record["trace_type"] for record in step_records)),
        "label_distribution": dict(Counter(record["label"] for record in step_records)),
        "reward_bucket_distribution": dict(Counter(record["reward_bucket"] for record in step_records)),
        "estimated_tokens_total": sum(
            estimated_tokens(record["prompt"] + " " + " ".join(record["previous_steps"]) + " " + record["current_step"])
            for record in step_records
        ),
    }

    write_jsonl(step_records, FINAL_FILE)
    write_jsonl(train_records, TRAIN_FILE)
    write_jsonl(val_records, VAL_FILE)
    write_jsonl(smoke_records, SMOKE_FILE)
    write_json(manifest, MANIFEST_FILE)
    print("✅ P6 PRM 训练数据组织完成。")
    print(manifest)


if __name__ == "__main__":
    main()
