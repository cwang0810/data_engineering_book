from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pipeline_utils import PROCESSED_DIR, QA_VIZ_DIR, ROOT_DIR, TRAINING_DIR, load_json, load_jsonl

SCRIPTS = sorted((ROOT_DIR / "src").glob("*.py"))
REQUIRED_FILES = [
    PROCESSED_DIR / "asset_manifest.jsonl",
    PROCESSED_DIR / "asset_collection_summary.json",
    PROCESSED_DIR / "llava_instruct.jsonl",
    PROCESSED_DIR / "llava_alignment.jsonl",
    PROCESSED_DIR / "llava_interleaved.jsonl",
    PROCESSED_DIR / "quality_audit.jsonl",
    PROCESSED_DIR / "low_quality_flags.jsonl",
    PROCESSED_DIR / "qa_visual_audit.jsonl",
    TRAINING_DIR / "final_llava_dataset.jsonl",
    TRAINING_DIR / "train.jsonl",
    TRAINING_DIR / "val.jsonl",
    TRAINING_DIR / "smoke_test.jsonl",
    TRAINING_DIR / "training_manifest.json",
    ROOT_DIR / "data" / "reports" / "p3_metrics.json",
    ROOT_DIR / "data" / "reports" / "p3_report.md",
]
RESULTS_FILE = ROOT_DIR / "data" / "reports" / "p3_test_results.json"
REPORT_FILE = ROOT_DIR / "data" / "reports" / "p3_test_report.md"


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

    asset_manifest = load_jsonl(PROCESSED_DIR / "asset_manifest.jsonl")
    alignment = load_jsonl(PROCESSED_DIR / "llava_alignment.jsonl")
    interleaved = load_jsonl(PROCESSED_DIR / "llava_interleaved.jsonl")
    train_records = load_jsonl(TRAINING_DIR / "train.jsonl")
    val_records = load_jsonl(TRAINING_DIR / "val.jsonl")
    smoke_records = load_jsonl(TRAINING_DIR / "smoke_test.jsonl")
    manifest = load_json(TRAINING_DIR / "training_manifest.json")
    low_quality = load_jsonl(PROCESSED_DIR / "low_quality_flags.jsonl")
    qa_visuals = list(QA_VIZ_DIR.glob("viz_*"))

    dataset_checks = [
        {
            "name": "required_files_exist",
            "passed": all(path.exists() for path in REQUIRED_FILES),
            "details": {
                "missing_files": [str(path.relative_to(ROOT_DIR)) for path in REQUIRED_FILES if not path.exists()]
            },
        },
        {
            "name": "asset_types_covered",
            "passed": {"general_image", "document_image", "chart_image"} <= {record["asset_type"] for record in asset_manifest},
            "details": {"asset_type_distribution": dict(Counter(record["asset_type"] for record in asset_manifest))},
        },
        {
            "name": "alignment_has_bboxes",
            "passed": all(len(record.get("bbox", [])) == 4 for record in alignment) and len(alignment) > 0,
            "details": {"alignment_records": len(alignment)},
        },
        {
            "name": "interleaved_is_multi_image",
            "passed": all(isinstance(record["image"], list) and len(record["image"]) == 2 for record in interleaved) and len(interleaved) > 0,
            "details": {"interleaved_records": len(interleaved)},
        },
        {
            "name": "train_val_no_overlap",
            "passed": not ({record["id"] for record in train_records} & {record["id"] for record in val_records}),
            "details": {"overlap": len({record["id"] for record in train_records} & {record["id"] for record in val_records})},
        },
        {
            "name": "smoke_covers_multiple_tasks",
            "passed": len({record["task_type"] for record in smoke_records}) >= 4,
            "details": {"smoke_task_distribution": dict(Counter(record["task_type"] for record in smoke_records))},
        },
        {
            "name": "low_quality_records_removed",
            "passed": len(low_quality) == 0,
            "details": {"low_quality_records": len(low_quality)},
        },
        {
            "name": "qa_visualizations_exist",
            "passed": len(qa_visuals) > 0,
            "details": {"qa_visualizations": len(qa_visuals)},
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
        "# P3 Test Report",
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
