from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAINING_DIR = PROJECT_ROOT / "data" / "training"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

RESULTS_JSON = REPORTS_DIR / "p2_test_results.json"
RESULTS_MD = REPORTS_DIR / "p2_test_report.md"

SCRIPT_FILES = sorted(path for path in SRC_DIR.glob("*.py"))
REQUIRED_FILES = [
    PROCESSED_DIR / "raw_chunks.jsonl",
    PROCESSED_DIR / "legal_seed_dataset.jsonl",
    PROCESSED_DIR / "instruction_taxonomy.json",
    PROCESSED_DIR / "domain_expert_sft.jsonl",
    PROCESSED_DIR / "synthetic_candidates_rejected.jsonl",
    PROCESSED_DIR / "legal_preference_pairs.jsonl",
    PROCESSED_DIR / "legal_qa_review.jsonl",
    PROCESSED_DIR / "legal_risk_refusal_sft.jsonl",
    PROCESSED_DIR / "legal_risk_register.jsonl",
    TRAINING_DIR / "final_sft_dataset.jsonl",
    TRAINING_DIR / "train.jsonl",
    TRAINING_DIR / "val.jsonl",
    TRAINING_DIR / "smoke_test.jsonl",
    TRAINING_DIR / "training_manifest.json",
    REPORTS_DIR / "p2_downstream_validation.json",
    REPORTS_DIR / "p2_downstream_validation.md",
    REPORTS_DIR / "p2_metrics.json",
    REPORTS_DIR / "p2_report.md",
]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_command(name: str, cmd: list[str]) -> dict:
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
    return {
        "name": name,
        "command": cmd,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def render_markdown(results: dict) -> str:
    lines = [
        "# P2 Test Report",
        "",
        f"- Timestamp: {results['timestamp_utc']}",
        f"- Overall status: {'PASS' if results['overall_passed'] else 'FAIL'}",
        f"- Passed checks: {results['passed_checks']}/{results['total_checks']}",
        "",
        "## Command Checks",
        "",
    ]
    for check in results["command_checks"]:
        lines.append(f"- {check['name']}: {'PASS' if check['passed'] else 'FAIL'}")
        lines.append(f"  - Command: `{' '.join(check['command'])}`")
        if check["stdout"]:
            lines.append(f"  - Stdout: `{check['stdout'][:400]}`")
        if check["stderr"]:
            lines.append(f"  - Stderr: `{check['stderr'][:400]}`")

    lines.extend(["", "## Dataset Checks", ""])
    for check in results["dataset_checks"]:
        lines.append(f"- {check['name']}: {'PASS' if check['passed'] else 'FAIL'}")
        lines.append(f"  - Details: `{json.dumps(check['details'], ensure_ascii=False)}`")

    return "\n".join(lines) + "\n"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    command_checks = [
        run_command("py_compile", [sys.executable, "-m", "py_compile", *[str(path) for path in SCRIPT_FILES]]),
        run_command("downstream_validation", [sys.executable, str(SRC_DIR / "downstream_validation.py")]),
        run_command("evaluate_factory", [sys.executable, str(SRC_DIR / "evaluate_factory.py")]),
    ]

    dataset_checks = []
    missing_files = [str(path.relative_to(PROJECT_ROOT)) for path in REQUIRED_FILES if not path.exists()]
    dataset_checks.append(
        {"name": "required_files_exist", "passed": not missing_files, "details": {"missing_files": missing_files}}
    )

    if not missing_files:
        seeds = load_jsonl(PROCESSED_DIR / "legal_seed_dataset.jsonl")
        accepted = load_jsonl(PROCESSED_DIR / "domain_expert_sft.jsonl")
        rejected = load_jsonl(PROCESSED_DIR / "synthetic_candidates_rejected.jsonl")
        preferences = load_jsonl(PROCESSED_DIR / "legal_preference_pairs.jsonl")
        qa_reviews = load_jsonl(PROCESSED_DIR / "legal_qa_review.jsonl")
        refusals = load_jsonl(PROCESSED_DIR / "legal_risk_refusal_sft.jsonl")
        final_dataset = load_jsonl(TRAINING_DIR / "final_sft_dataset.jsonl")
        train = load_jsonl(TRAINING_DIR / "train.jsonl")
        val = load_jsonl(TRAINING_DIR / "val.jsonl")
        smoke = load_jsonl(TRAINING_DIR / "smoke_test.jsonl")
        manifest = json.loads((TRAINING_DIR / "training_manifest.json").read_text(encoding="utf-8"))
        taxonomy = json.loads((PROCESSED_DIR / "instruction_taxonomy.json").read_text(encoding="utf-8"))
        downstream_validation = json.loads((REPORTS_DIR / "p2_downstream_validation.json").read_text(encoding="utf-8"))
        downstream_summary = downstream_validation["summary"]

        dataset_checks.extend(
            [
                {
                    "name": "seed_count_positive",
                    "passed": len(seeds) > 0,
                    "details": {"seed_count": len(seeds)},
                },
                {
                    "name": "accepted_count_matches_seed_x_tasks",
                    "passed": len(accepted) == len(seeds) * 3,
                    "details": {"accepted": len(accepted), "expected": len(seeds) * 3},
                },
                {
                    "name": "preference_pairs_cover_accepted",
                    "passed": len(preferences) == len(accepted),
                    "details": {"preference_pairs": len(preferences), "accepted": len(accepted)},
                },
                {
                    "name": "qa_reviews_cover_accepted",
                    "passed": len(qa_reviews) == len(accepted),
                    "details": {"qa_reviews": len(qa_reviews), "accepted": len(accepted)},
                },
                {
                    "name": "train_val_no_overlap",
                    "passed": not ({item["id"] for item in train} & {item["id"] for item in val}),
                    "details": {"overlap": len({item['id'] for item in train} & {item['id'] for item in val})},
                },
                {
                    "name": "smoke_covers_multiple_tasks",
                    "passed": len({item["task_type"] for item in smoke}) >= 3,
                    "details": {"smoke_task_distribution": dict(Counter(item["task_type"] for item in smoke))},
                },
                {
                    "name": "risk_refusals_present",
                    "passed": len(refusals) >= 4,
                    "details": {"risk_refusals": len(refusals)},
                },
                {
                    "name": "manifest_matches_final_dataset",
                    "passed": manifest["num_records"] == len(final_dataset),
                    "details": {"manifest": manifest["num_records"], "actual": len(final_dataset)},
                },
                {
                    "name": "taxonomy_has_three_tasks",
                    "passed": len(taxonomy.get("target_tasks", [])) == 3,
                    "details": {"target_tasks": taxonomy.get("target_tasks", [])},
                },
                {
                    "name": "rejected_candidates_present",
                    "passed": len(rejected) >= len(accepted),
                    "details": {"rejected": len(rejected), "accepted": len(accepted)},
                },
                {
                    "name": "downstream_validation_sample_size",
                    "passed": downstream_summary["sample_size"] == min(50, len(preferences)),
                    "details": {
                        "sample_size": downstream_summary["sample_size"],
                        "expected": min(50, len(preferences)),
                    },
                },
                {
                    "name": "downstream_validation_covers_all_tasks",
                    "passed": len(downstream_summary["task_distribution"]) == 3,
                    "details": {"task_distribution": downstream_summary["task_distribution"]},
                },
                {
                    "name": "downstream_validation_chosen_wins",
                    "passed": downstream_summary["chosen_win_rate"] >= 0.95,
                    "details": {
                        "chosen_win_rate": downstream_summary["chosen_win_rate"],
                        "chosen_avg_total_score": downstream_summary["chosen_avg_total_score"],
                        "rejected_avg_total_score": downstream_summary["rejected_avg_total_score"],
                    },
                },
            ]
        )

    all_checks = command_checks + dataset_checks
    results = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overall_passed": all(check["passed"] for check in all_checks),
        "total_checks": len(all_checks),
        "passed_checks": sum(1 for check in all_checks if check["passed"]),
        "command_checks": command_checks,
        "dataset_checks": dataset_checks,
    }

    RESULTS_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    RESULTS_MD.write_text(render_markdown(results), encoding="utf-8")

    print(json.dumps(results, ensure_ascii=False, indent=2))
    if not results["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
