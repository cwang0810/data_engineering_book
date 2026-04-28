from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone

from pipeline_utils import BOOKS_DIR, PROCESSED_DIR, REPORTS_DIR, ROOT_DIR, TRAINING_DIR, load_json, load_jsonl

SCRIPTS = sorted((ROOT_DIR / "src").glob("*.py"))
REQUIRED_FILES = [
    PROCESSED_DIR / "seed_pool.jsonl",
    PROCESSED_DIR / "chapter_plan.json",
    PROCESSED_DIR / "synthetic_textbook_chapters.jsonl",
    PROCESSED_DIR / "verified_textbook.jsonl",
    PROCESSED_DIR / "verification_failures.jsonl",
    PROCESSED_DIR / "execution_results.jsonl",
    PROCESSED_DIR / "quality_audit.jsonl",
    PROCESSED_DIR / "low_quality_flags.jsonl",
    PROCESSED_DIR / "manual_review_samples.jsonl",
    PROCESSED_DIR / "curriculum_map.json",
    PROCESSED_DIR / "textbook_catalog.json",
    PROCESSED_DIR / "editorial_style_guide.md",
    TRAINING_DIR / "final_textbook_dataset.jsonl",
    TRAINING_DIR / "train.jsonl",
    TRAINING_DIR / "val.jsonl",
    TRAINING_DIR / "smoke_test.jsonl",
    TRAINING_DIR / "training_manifest.json",
    REPORTS_DIR / "p4_metrics.json",
    REPORTS_DIR / "p4_report.md",
    BOOKS_DIR / "foundations_of_quantitative_reasoning.md",
    BOOKS_DIR / "python_problem_solving_workbook.md",
    BOOKS_DIR / "teacher_guide.md",
]
RESULTS_FILE = REPORTS_DIR / "p4_test_results.json"
REPORT_FILE = REPORTS_DIR / "p4_test_report.md"


def run_command(command: list[str]) -> dict:
    completed = subprocess.run(command, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def main() -> None:
    command_checks = []
    py_compile = run_command([sys.executable, "-m", "py_compile", *[str(path) for path in SCRIPTS]])
    py_compile["name"] = "py_compile"
    command_checks.append(py_compile)

    evaluate = run_command([sys.executable, str(ROOT_DIR / "src" / "evaluate_factory.py")])
    evaluate["name"] = "evaluate_factory"
    command_checks.append(evaluate)

    verified = load_jsonl(PROCESSED_DIR / "verified_textbook.jsonl")
    failures = load_jsonl(PROCESSED_DIR / "verification_failures.jsonl")
    smoke_records = load_jsonl(TRAINING_DIR / "smoke_test.jsonl")
    train_records = load_jsonl(TRAINING_DIR / "train.jsonl")
    val_records = load_jsonl(TRAINING_DIR / "val.jsonl")
    manifest = load_json(TRAINING_DIR / "training_manifest.json")
    quality = load_json(PROCESSED_DIR / "quality_summary.json")
    curriculum = load_json(PROCESSED_DIR / "curriculum_map.json")

    dataset_checks = [
        {
            "name": "required_files_exist",
            "passed": all(path.exists() for path in REQUIRED_FILES),
            "details": {"missing_files": [str(path.relative_to(ROOT_DIR)) for path in REQUIRED_FILES if not path.exists()]},
        },
        {
            "name": "both_domains_present",
            "passed": {"math", "code"} <= {record["domain"] for record in verified},
            "details": {"domain_distribution": dict(Counter(record["domain"] for record in verified))},
        },
        {
            "name": "curriculum_has_two_volumes",
            "passed": len(curriculum["volumes"]) == 2,
            "details": {"volumes": curriculum["volumes"]},
        },
        {
            "name": "verification_pass_rate_reasonable",
            "passed": len(verified) >= 0.8 * (len(verified) + len(failures)),
            "details": {"verified": len(verified), "failed": len(failures)},
        },
        {
            "name": "train_val_no_overlap",
            "passed": not ({record["id"] for record in train_records} & {record["id"] for record in val_records}),
            "details": {"overlap": len({record["id"] for record in train_records} & {record["id"] for record in val_records})},
        },
        {
            "name": "smoke_covers_both_domains",
            "passed": {"math", "code"} <= {record["domain"] for record in smoke_records},
            "details": {"smoke_domain_distribution": dict(Counter(record["domain"] for record in smoke_records))},
        },
        {
            "name": "quality_has_no_failures",
            "passed": quality["failed_records"] == 0,
            "details": quality,
        },
        {
            "name": "manifest_matches_final_count",
            "passed": manifest["num_train_records"] + manifest["num_val_records"] == manifest["num_records"],
            "details": manifest,
        },
    ]

    results = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overall_passed": all(item["passed"] for item in command_checks) and all(item["passed"] for item in dataset_checks),
        "total_checks": len(command_checks) + len(dataset_checks),
        "passed_checks": sum(item["passed"] for item in command_checks) + sum(item["passed"] for item in dataset_checks),
        "command_checks": command_checks,
        "dataset_checks": dataset_checks,
    }

    RESULTS_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# P4 Test Report",
        "",
        f"- Timestamp: {results['timestamp_utc']}",
        f"- Overall status: {'PASS' if results['overall_passed'] else 'FAIL'}",
        f"- Passed checks: {results['passed_checks']}/{results['total_checks']}",
        "",
        "## Command Checks",
        "",
    ]
    for check in command_checks:
        lines.append(f"- {check['name']}: {'PASS' if check['passed'] else 'FAIL'}")
        lines.append(f"  - Command: `{' '.join(check['command'])}`")
        if check["stdout"]:
            lines.append(f"  - Stdout: `{check['stdout'][:600]}`")
        if check["stderr"]:
            lines.append(f"  - Stderr: `{check['stderr'][:600]}`")

    lines.extend(["", "## Dataset Checks", ""])
    for check in dataset_checks:
        lines.append(f"- {check['name']}: {'PASS' if check['passed'] else 'FAIL'}")
        lines.append(f"  - Details: `{json.dumps(check['details'], ensure_ascii=False)}`")

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
