from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, UTC
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TRAINING_DIR = DATA_DIR / "training"
REPORT_DIR = DATA_DIR / "reports"

RESULTS_JSON = REPORT_DIR / "p1_test_results.json"
RESULTS_MD = REPORT_DIR / "p1_test_report.md"

SCRIPT_FILES = [
    CURRENT_DIR / "pipeline_utils.py",
    CURRENT_DIR / "1_download_data.py",
    CURRENT_DIR / "2_process_warc.py",
    CURRENT_DIR / "3_clean_data.py",
    CURRENT_DIR / "4_deduplicate.py",
    CURRENT_DIR / "5_split_lang.py",
    CURRENT_DIR / "6_quality_filter.py",
    CURRENT_DIR / "7_prepare_training_data.py",
    CURRENT_DIR / "8_evaluate_dataset.py",
    CURRENT_DIR / "9_training_smoke_test.py",
    CURRENT_DIR / "10_run_p1_checks.py",
    CURRENT_DIR / "test_dedup_check.py",
]

REQUIRED_FILES = [
    PROCESSED_DIR / "extracted_data.jsonl",
    PROCESSED_DIR / "clean_data.jsonl",
    PROCESSED_DIR / "deduplicated_data.jsonl",
    PROCESSED_DIR / "data_en.jsonl",
    PROCESSED_DIR / "data_zh.jsonl",
    PROCESSED_DIR / "final_data_en.jsonl",
    PROCESSED_DIR / "final_data_zh.jsonl",
    PROCESSED_DIR / "final_data.jsonl",
    PROCESSED_DIR / "quality_filter_stats.json",
    TRAINING_DIR / "serialized_dataset.jsonl",
    TRAINING_DIR / "train.jsonl",
    TRAINING_DIR / "val.jsonl",
    TRAINING_DIR / "smoke_test.jsonl",
    TRAINING_DIR / "training_manifest.json",
    REPORT_DIR / "p1_metrics.json",
    REPORT_DIR / "p1_report.md",
]


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def run_command(name: str, cmd: list[str], cwd: Path) -> dict:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "name": name,
        "command": cmd,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def build_command_checks() -> list[dict]:
    compile_cmd = [sys.executable, "-m", "py_compile", *[str(path) for path in SCRIPT_FILES]]
    checks = [
        run_command("py_compile", compile_cmd, PROJECT_ROOT),
        run_command("dedup_unit_check", [sys.executable, str(CURRENT_DIR / "test_dedup_check.py")], PROJECT_ROOT),
        run_command("training_smoke_test", [sys.executable, str(CURRENT_DIR / "9_training_smoke_test.py")], PROJECT_ROOT),
        run_command("dataset_evaluation", [sys.executable, str(CURRENT_DIR / "8_evaluate_dataset.py")], PROJECT_ROOT),
    ]
    return checks


def build_dataset_checks() -> list[dict]:
    checks = []

    missing_files = [str(path.relative_to(PROJECT_ROOT)) for path in REQUIRED_FILES if not path.exists()]
    checks.append(
        {
            "name": "required_files_exist",
            "passed": not missing_files,
            "details": {"missing_files": missing_files},
        }
    )

    if missing_files:
        return checks

    manifest = load_json(TRAINING_DIR / "training_manifest.json")
    metrics = load_json(REPORT_DIR / "p1_metrics.json")
    quality_stats = load_json(PROCESSED_DIR / "quality_filter_stats.json")

    final_records = load_jsonl(PROCESSED_DIR / "final_data.jsonl")
    final_en_records = load_jsonl(PROCESSED_DIR / "final_data_en.jsonl")
    final_zh_records = load_jsonl(PROCESSED_DIR / "final_data_zh.jsonl")
    train_records = load_jsonl(TRAINING_DIR / "train.jsonl")
    val_records = load_jsonl(TRAINING_DIR / "val.jsonl")
    smoke_records = load_jsonl(TRAINING_DIR / "smoke_test.jsonl")

    final_langs = Counter(record.get("lang", "unknown") for record in final_records)
    train_hashes = {record["text_sha1"] for record in train_records}
    val_hashes = {record["text_sha1"] for record in val_records}
    smoke_ids = {record["id"] for record in smoke_records}
    train_ids = {record["id"] for record in train_records}
    all_final_hashes = [record["text_sha1"] for record in final_records]

    checks.append(
        {
            "name": "final_file_count_matches_language_splits",
            "passed": len(final_records) == len(final_en_records) + len(final_zh_records),
            "details": {
                "final": len(final_records),
                "final_en": len(final_en_records),
                "final_zh": len(final_zh_records),
            },
        }
    )
    checks.append(
        {
            "name": "training_manifest_counts_match_files",
            "passed": (
                manifest.get("num_records") == len(final_records)
                and manifest.get("num_train_records") == len(train_records)
                and manifest.get("num_val_records") == len(val_records)
                and manifest.get("num_smoke_test_records") == len(smoke_records)
            ),
            "details": {
                "manifest_num_records": manifest.get("num_records"),
                "actual_num_records": len(final_records),
                "manifest_num_train_records": manifest.get("num_train_records"),
                "actual_num_train_records": len(train_records),
                "manifest_num_val_records": manifest.get("num_val_records"),
                "actual_num_val_records": len(val_records),
                "manifest_num_smoke_test_records": manifest.get("num_smoke_test_records"),
                "actual_num_smoke_test_records": len(smoke_records),
            },
        }
    )
    checks.append(
        {
            "name": "train_val_no_overlap",
            "passed": not (train_hashes & val_hashes),
            "details": {"overlap_count": len(train_hashes & val_hashes)},
        }
    )
    checks.append(
        {
            "name": "smoke_test_is_subset_of_train",
            "passed": smoke_ids <= train_ids,
            "details": {
                "smoke_records": len(smoke_ids),
                "missing_from_train": sorted(smoke_ids - train_ids)[:10],
            },
        }
    )
    checks.append(
        {
            "name": "final_dataset_no_exact_duplicates",
            "passed": len(all_final_hashes) == len(set(all_final_hashes)),
            "details": {
                "final_records": len(all_final_hashes),
                "unique_hashes": len(set(all_final_hashes)),
            },
        }
    )
    checks.append(
        {
            "name": "metrics_match_final_dataset",
            "passed": (
                metrics.get("stage_counts", {}).get("final") == len(final_records)
                and metrics.get("final_summary", {}).get("languages") == dict(final_langs)
            ),
            "details": {
                "metrics_final_count": metrics.get("stage_counts", {}).get("final"),
                "actual_final_count": len(final_records),
                "metrics_languages": metrics.get("final_summary", {}).get("languages"),
                "actual_languages": dict(final_langs),
            },
        }
    )
    checks.append(
        {
            "name": "quality_filter_stats_match_outputs",
            "passed": (
                quality_stats.get("en", {}).get("kept") == len(final_en_records)
                and quality_stats.get("zh", {}).get("kept") == len(final_zh_records)
            ),
            "details": {
                "quality_en_kept": quality_stats.get("en", {}).get("kept"),
                "actual_final_en": len(final_en_records),
                "quality_zh_kept": quality_stats.get("zh", {}).get("kept"),
                "actual_final_zh": len(final_zh_records),
            },
        }
    )
    checks.append(
        {
            "name": "smoke_test_covers_multiple_languages",
            "passed": len({record.get("lang", "unknown") for record in smoke_records}) >= 2,
            "details": {
                "smoke_languages": dict(Counter(record.get("lang", "unknown") for record in smoke_records))
            },
        }
    )
    checks.append(
        {
            "name": "stage_counts_are_monotonic",
            "passed": (
                count_lines(PROCESSED_DIR / "extracted_data.jsonl")
                >= count_lines(PROCESSED_DIR / "clean_data.jsonl")
                >= count_lines(PROCESSED_DIR / "deduplicated_data.jsonl")
                >= count_lines(PROCESSED_DIR / "final_data.jsonl")
            ),
            "details": {
                "extracted": count_lines(PROCESSED_DIR / "extracted_data.jsonl"),
                "clean": count_lines(PROCESSED_DIR / "clean_data.jsonl"),
                "deduplicated": count_lines(PROCESSED_DIR / "deduplicated_data.jsonl"),
                "final": count_lines(PROCESSED_DIR / "final_data.jsonl"),
            },
        }
    )

    return checks


def render_markdown(results: dict) -> str:
    lines = [
        "# P1 Test Report",
        "",
        f"- Timestamp: {results['timestamp_utc']}",
        f"- Overall status: {'PASS' if results['overall_passed'] else 'FAIL'}",
        f"- Passed checks: {results['passed_checks']}/{results['total_checks']}",
        "",
        "## Command Checks",
        "",
    ]

    for check in results["command_checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- {check['name']}: {status}")
        lines.append(f"  - Command: `{' '.join(check['command'])}`")
        if check["stdout"]:
            lines.append(f"  - Stdout: `{check['stdout'][:400]}`")
        if check["stderr"]:
            lines.append(f"  - Stderr: `{check['stderr'][:400]}`")

    lines.extend(["", "## Dataset Checks", ""])
    for check in results["dataset_checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- {check['name']}: {status}")
        lines.append(f"  - Details: `{json.dumps(check['details'], ensure_ascii=False)}`")

    lines.extend(["", "## Summary", ""])
    lines.append(f"- Final dataset size: {results['summary']['final_records']}")
    lines.append(f"- Train/val size: {results['summary']['train_records']}/{results['summary']['val_records']}")
    lines.append(f"- Smoke test size: {results['summary']['smoke_test_records']}")
    lines.append(f"- Final language distribution: `{json.dumps(results['summary']['final_languages'], ensure_ascii=False)}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    command_checks = build_command_checks()
    dataset_checks = build_dataset_checks()
    all_checks = command_checks + dataset_checks

    final_records = load_jsonl(PROCESSED_DIR / "final_data.jsonl") if (PROCESSED_DIR / "final_data.jsonl").exists() else []
    train_records = load_jsonl(TRAINING_DIR / "train.jsonl") if (TRAINING_DIR / "train.jsonl").exists() else []
    val_records = load_jsonl(TRAINING_DIR / "val.jsonl") if (TRAINING_DIR / "val.jsonl").exists() else []
    smoke_records = load_jsonl(TRAINING_DIR / "smoke_test.jsonl") if (TRAINING_DIR / "smoke_test.jsonl").exists() else []

    results = {
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overall_passed": all(check["passed"] for check in all_checks),
        "total_checks": len(all_checks),
        "passed_checks": sum(1 for check in all_checks if check["passed"]),
        "command_checks": command_checks,
        "dataset_checks": dataset_checks,
        "summary": {
            "final_records": len(final_records),
            "train_records": len(train_records),
            "val_records": len(val_records),
            "smoke_test_records": len(smoke_records),
            "final_languages": dict(Counter(record.get("lang", "unknown") for record in final_records)),
        },
    }

    with RESULTS_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with RESULTS_MD.open("w", encoding="utf-8") as f:
        f.write(render_markdown(results))

    print(json.dumps(results, ensure_ascii=False, indent=2))
    if not results["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
