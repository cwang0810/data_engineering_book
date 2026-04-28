from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone

from pipeline_utils import PROCESSED_DIR, REPORTS_DIR, ROOT_DIR, TRAINING_DIR, load_json, load_jsonl

SCRIPTS = sorted((ROOT_DIR / "src").glob("*.py"))
REQUIRED_FILES = [
    PROCESSED_DIR / "seed_pool.jsonl",
    PROCESSED_DIR / "task_spec.json",
    PROCESSED_DIR / "cot_traces.jsonl",
    PROCESSED_DIR / "trace_summary.json",
    PROCESSED_DIR / "validated_traces.jsonl",
    PROCESSED_DIR / "step_rewards.jsonl",
    PROCESSED_DIR / "validation_summary.json",
    TRAINING_DIR / "prm_step_dataset.jsonl",
    TRAINING_DIR / "train.jsonl",
    TRAINING_DIR / "val.jsonl",
    TRAINING_DIR / "smoke_test.jsonl",
    TRAINING_DIR / "training_manifest.json",
    REPORTS_DIR / "p6_metrics.json",
    REPORTS_DIR / "p6_report.md",
]
RESULTS_FILE = REPORTS_DIR / "p6_test_results.json"
REPORT_FILE = REPORTS_DIR / "p6_test_report.md"


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

    evaluate = run_command([sys.executable, str(ROOT_DIR / "src" / "evaluate_prm.py")])
    evaluate["name"] = "evaluate_prm"
    command_checks.append(evaluate)

    traces = load_jsonl(PROCESSED_DIR / "validated_traces.jsonl")
    steps = load_jsonl(TRAINING_DIR / "prm_step_dataset.jsonl")
    smoke = load_jsonl(TRAINING_DIR / "smoke_test.jsonl")
    train = load_jsonl(TRAINING_DIR / "train.jsonl")
    val = load_jsonl(TRAINING_DIR / "val.jsonl")
    manifest = load_json(TRAINING_DIR / "training_manifest.json")

    dataset_checks = [
        {
            "name": "required_files_exist",
            "passed": all(path.exists() for path in REQUIRED_FILES),
            "details": {"missing_files": [str(path.relative_to(ROOT_DIR)) for path in REQUIRED_FILES if not path.exists()]},
        },
        {
            "name": "both_domains_present",
            "passed": {"math", "code"} <= {trace["domain"] for trace in traces},
            "details": {"domain_distribution": dict(Counter(trace["domain"] for trace in traces))},
        },
        {
            "name": "trace_types_present",
            "passed": {"positive", "negative", "repair"} <= {trace["trace_type"] for trace in traces},
            "details": {"trace_type_distribution": dict(Counter(trace["trace_type"] for trace in traces))},
        },
        {
            "name": "step_labels_cover_both_classes",
            "passed": {0, 1} <= {step["label"] for step in steps},
            "details": {"label_distribution": dict(Counter(step["label"] for step in steps))},
        },
        {
            "name": "reward_buckets_present",
            "passed": len({trace["reward_bucket"] for trace in traces}) >= 3,
            "details": {"reward_bucket_distribution": dict(Counter(trace["reward_bucket"] for trace in traces))},
        },
        {
            "name": "train_val_no_overlap",
            "passed": not ({record["record_id"] for record in train} & {record["record_id"] for record in val}),
            "details": {"overlap": len({record["record_id"] for record in train} & {record["record_id"] for record in val})},
        },
        {
            "name": "smoke_covers_both_domains",
            "passed": {"math", "code"} <= {record["domain"] for record in smoke},
            "details": {"smoke_domain_distribution": dict(Counter(record["domain"] for record in smoke))},
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
        "# P6 Test Report",
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
