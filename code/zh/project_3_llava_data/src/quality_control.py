from __future__ import annotations

from collections import Counter

from pipeline_utils import DATA_DIR, PROCESSED_DIR, parse_bbox_mentions, load_jsonl, write_json, write_jsonl

INPUT_FILES = {
    "llava_instruct": PROCESSED_DIR / "llava_instruct.jsonl",
    "llava_alignment": PROCESSED_DIR / "llava_alignment.jsonl",
    "llava_interleaved": PROCESSED_DIR / "llava_interleaved.jsonl",
}
AUDIT_FILE = PROCESSED_DIR / "quality_audit.jsonl"
LOW_QUALITY_FILE = PROCESSED_DIR / "low_quality_flags.jsonl"
SUMMARY_FILE = PROCESSED_DIR / "quality_summary.json"
MANUAL_REVIEW_FILE = PROCESSED_DIR / "manual_review_samples.jsonl"


def validate_record(dataset_name: str, record: dict) -> tuple[dict, dict | None]:
    flags: list[str] = []
    image_field = record["image"]
    image_paths = image_field if isinstance(image_field, list) else [image_field]
    missing = [path for path in image_paths if not (DATA_DIR / path).exists()]
    if missing:
        flags.append("missing_image")

    conversations = record.get("conversations", [])
    if len(conversations) < 2:
        flags.append("too_few_turns")
    if conversations and "<image>" not in conversations[0]["value"]:
        flags.append("missing_image_token")

    for turn_index, turn in enumerate(conversations):
        if not turn.get("value", "").strip():
            flags.append(f"empty_turn_{turn_index}")

    if dataset_name == "llava_alignment":
        bbox = record.get("bbox")
        if not bbox or len(bbox) != 4:
            flags.append("missing_bbox")
        else:
            ymin, xmin, ymax, xmax = bbox
            if not (0 <= ymin <= ymax <= 1000 and 0 <= xmin <= xmax <= 1000):
                flags.append("bbox_out_of_range")
        answer_bboxes = parse_bbox_mentions(conversations[-1]["value"])
        if not answer_bboxes:
            flags.append("bbox_not_mentioned_in_answer")

    audit_record = {
        "dataset_name": dataset_name,
        "id": record["id"],
        "task_type": record["task_type"],
        "asset_type": record["asset_type"],
        "passed": not flags,
        "flags": flags,
    }
    low_quality = audit_record if flags else None
    return audit_record, low_quality


def main() -> None:
    audit_records: list[dict] = []
    low_quality_records: list[dict] = []
    manual_review_records: list[dict] = []

    for dataset_name, path in INPUT_FILES.items():
        records = load_jsonl(path)
        for record in records:
            audit_record, low_quality = validate_record(dataset_name, record)
            audit_records.append(audit_record)
            if low_quality:
                low_quality_records.append(low_quality)

        manual_review_records.extend(records[:4])

    summary = {
        "total_records": len(audit_records),
        "passed_records": sum(1 for item in audit_records if item["passed"]),
        "failed_records": len(low_quality_records),
        "dataset_distribution": dict(Counter(item["dataset_name"] for item in audit_records)),
        "task_distribution": dict(Counter(item["task_type"] for item in audit_records)),
    }

    write_jsonl(audit_records, AUDIT_FILE)
    write_jsonl(low_quality_records, LOW_QUALITY_FILE)
    write_jsonl(manual_review_records, MANUAL_REVIEW_FILE)
    write_json(summary, SUMMARY_FILE)
    print("✅ 质量控制与抽检规则执行完成。")
    print(summary)


if __name__ == "__main__":
    main()
