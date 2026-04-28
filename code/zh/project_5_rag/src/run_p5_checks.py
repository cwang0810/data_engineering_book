from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone

from pipeline_utils import EVAL_DIR, PROCESSED_DIR, REPORTS_DIR, ROOT_DIR, load_json, load_jsonl

SCRIPTS = sorted((ROOT_DIR / "src").glob("*.py"))
REQUIRED_FILES = [
    PROCESSED_DIR / "page_units.jsonl",
    PROCESSED_DIR / "block_units.jsonl",
    PROCESSED_DIR / "rag_index.json",
    EVAL_DIR / "reference_questions.jsonl",
    EVAL_DIR / "evaluation_results.jsonl",
    EVAL_DIR / "failure_replay.jsonl",
    REPORTS_DIR / "p5_metrics.json",
    REPORTS_DIR / "p5_report.md",
]
RESULTS_FILE = REPORTS_DIR / "p5_test_results.json"
REPORT_FILE = REPORTS_DIR / "p5_test_report.md"


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

    evaluate = run_command([sys.executable, str(ROOT_DIR / "src" / "evaluate_rag.py")])
    evaluate["name"] = "evaluate_rag"
    command_checks.append(evaluate)

    metrics = load_json(REPORTS_DIR / "p5_metrics.json")
    pages = load_jsonl(PROCESSED_DIR / "page_units.jsonl")
    blocks = load_jsonl(PROCESSED_DIR / "block_units.jsonl")
    eval_results = load_jsonl(EVAL_DIR / "evaluation_results.jsonl")

    dataset_checks = [
        {
            "name": "required_files_exist",
            "passed": all(path.exists() for path in REQUIRED_FILES),
            "details": {"missing_files": [str(path.relative_to(ROOT_DIR)) for path in REQUIRED_FILES if not path.exists()]},
        },
        {
            "name": "page_index_non_empty",
            "passed": len(pages) > 100 and len(blocks) > len(pages),
            "details": {"num_pages": len(pages), "num_blocks": len(blocks)},
        },
        {
            "name": "retrieval_hit_rate_reasonable",
            "passed": metrics["retrieval_hit_rate_at_4"] >= 0.75,
            "details": metrics,
        },
        {
            "name": "citation_accuracy_reasonable",
            "passed": metrics["citation_accuracy"] >= 0.75,
            "details": metrics,
        },
        {
            "name": "all_eval_answers_non_empty",
            "passed": all(item["answer"].strip() for item in eval_results),
            "details": {"num_eval_results": len(eval_results)},
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
        "# P5 Test Report",
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
