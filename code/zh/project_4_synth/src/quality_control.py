from __future__ import annotations

from collections import Counter

from pipeline_utils import PROCESSED_DIR, estimated_tokens, load_jsonl, normalize_text, write_json, write_jsonl

INPUT_FILE = PROCESSED_DIR / "verified_textbook.jsonl"
AUDIT_FILE = PROCESSED_DIR / "quality_audit.jsonl"
LOW_QUALITY_FILE = PROCESSED_DIR / "low_quality_flags.jsonl"
SUMMARY_FILE = PROCESSED_DIR / "quality_summary.json"
MANUAL_REVIEW_FILE = PROCESSED_DIR / "manual_review_samples.jsonl"


def validate_record(record: dict, seen_hashes: set[str]) -> tuple[dict, dict | None]:
    flags: list[str] = []
    if not record.get("chapter_title"):
        flags.append("missing_title")
    if not record.get("lesson_text"):
        flags.append("missing_lesson_text")
    if not record.get("chapter_markdown"):
        flags.append("missing_markdown")
    if not record.get("learning_objectives"):
        flags.append("missing_learning_objectives")
    if not record.get("end_of_chapter_checks"):
        flags.append("missing_end_of_chapter_checks")
    if record["domain"] == "math" and not record.get("exercise_answer"):
        flags.append("missing_math_answer")
    if record["domain"] == "code" and not record.get("unit_tests"):
        flags.append("missing_unit_tests")
    if not record.get("verification", {}).get("passed"):
        flags.append("verification_failed")
    if estimated_tokens(record.get("chapter_markdown", "")) < 60:
        flags.append("chapter_too_short")

    fingerprint = normalize_text(record.get("chapter_markdown", ""))
    if fingerprint in seen_hashes:
        flags.append("duplicate_chapter")
    else:
        seen_hashes.add(fingerprint)

    audit = {
        "id": record["id"],
        "domain": record["domain"],
        "topic": record["topic"],
        "difficulty": record["difficulty"],
        "passed": not flags,
        "flags": flags,
    }
    return audit, (audit if flags else None)


def main() -> None:
    records = load_jsonl(INPUT_FILE)
    seen_hashes: set[str] = set()
    audits: list[dict] = []
    low_quality: list[dict] = []

    for record in records:
        audit, maybe_low = validate_record(record, seen_hashes)
        audits.append(audit)
        if maybe_low:
            low_quality.append(maybe_low)

    summary = {
        "total_records": len(audits),
        "passed_records": sum(1 for audit in audits if audit["passed"]),
        "failed_records": len(low_quality),
        "domain_distribution": dict(Counter(record["domain"] for record in records)),
        "topic_distribution": dict(Counter(record["topic"] for record in records)),
        "difficulty_distribution": dict(Counter(record["difficulty"] for record in records)),
    }

    write_jsonl(audits, AUDIT_FILE)
    write_jsonl(low_quality, LOW_QUALITY_FILE)
    write_jsonl(records[:8], MANUAL_REVIEW_FILE)
    write_json(summary, SUMMARY_FILE)
    print("✅ 质量风控与人工抽检样本生成完成。")
    print(summary)


if __name__ == "__main__":
    main()
