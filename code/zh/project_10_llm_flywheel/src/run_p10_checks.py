from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pipeline_utils import CONSOLE_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_standard_dirs, load_json, load_jsonl, write_json

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
RESULTS_FILE = REPORTS_DIR / "p10_test_results.json"
REPORT_FILE = REPORTS_DIR / "p10_test_report.md"


def run_command(command: list[str], name: str) -> dict:
    result = subprocess.run(command, capture_output=True, text=True)
    return {
        "name": name,
        "command": command,
        "returncode": result.returncode,
        "passed": result.returncode == 0,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def main() -> None:
    ensure_standard_dirs()
    registry = load_json(PROCESSED_DIR / "upstream_project_registry.json")
    phase_inventory = load_json(PROCESSED_DIR / "phase_inventory.json")
    architecture = load_json(PROCESSED_DIR / "flywheel_architecture.json")
    boundaries = load_json(PROCESSED_DIR / "system_boundaries.json")
    stage_plan = load_json(PROCESSED_DIR / "stage_plan.json")
    runs = load_jsonl(PROCESSED_DIR / "flywheel_runs.jsonl")
    milestones = load_json(CONSOLE_DIR / "milestone_board.json")
    bottlenecks = load_json(PROCESSED_DIR / "bottleneck_analysis.json")
    cost_model = load_json(PROCESSED_DIR / "cost_model.json")
    org_model = load_json(PROCESSED_DIR / "org_operating_model.json")
    dashboard = load_json(CONSOLE_DIR / "executive_dashboard.json")
    metrics = load_json(REPORTS_DIR / "p10_metrics.json")

    py_files = sorted(str(path) for path in SRC_DIR.glob("*.py"))
    command_checks = [
        run_command([sys.executable, "-m", "py_compile", *py_files], "py_compile"),
        run_command([sys.executable, str(SRC_DIR / "evaluate_flywheel.py")], "evaluate_flywheel"),
    ]

    dataset_checks = []
    dataset_checks.append(
        {
            "name": "required_files_exist",
            "passed": all(
                path.exists()
                for path in [
                    PROCESSED_DIR / "upstream_project_registry.json",
                    PROCESSED_DIR / "phase_inventory.json",
                    PROCESSED_DIR / "flywheel_architecture.json",
                    PROCESSED_DIR / "system_boundaries.json",
                    PROCESSED_DIR / "stage_plan.json",
                    PROCESSED_DIR / "flywheel_runs.jsonl",
                    PROCESSED_DIR / "bottleneck_analysis.json",
                    PROCESSED_DIR / "cost_model.json",
                    PROCESSED_DIR / "org_operating_model.json",
                    CONSOLE_DIR / "milestone_board.json",
                    CONSOLE_DIR / "executive_dashboard.json",
                ]
            ),
            "details": {},
        }
    )
    dataset_checks.append(
        {
            "name": "all_upstream_projects_registered",
            "passed": len(registry) == 9 and all(item["overall_passed"] for item in registry),
            "details": {"project_count": len(registry)},
        }
    )
    dataset_checks.append(
        {
            "name": "phase_inventory_consistent",
            "passed": phase_inventory["project_count"] == len(registry) and phase_inventory["all_projects_passed"],
            "details": phase_inventory,
        }
    )
    dataset_checks.append(
        {
            "name": "architecture_layers_and_control_points_present",
            "passed": len(architecture["layers"]) >= 5 and len(architecture["control_points"]) >= 4,
            "details": {"layer_count": len(architecture["layers"]), "control_point_count": len(architecture["control_points"])},
        }
    )
    dataset_checks.append(
        {
            "name": "stage_plan_covers_end_to_end",
            "passed": len(stage_plan) == 5 and all(len(item["projects"]) >= 1 for item in stage_plan),
            "details": {"stage_titles": [item["title"] for item in stage_plan]},
        }
    )
    dataset_checks.append(
        {
            "name": "flywheel_runs_complete",
            "passed": len(runs) == 5 and all(item["status"] == "completed" for item in runs),
            "details": {"run_count": len(runs)},
        }
    )
    dataset_checks.append(
        {
            "name": "milestones_complete",
            "passed": len(milestones) >= 5 and all(item["status"] == "done" for item in milestones),
            "details": {"milestone_count": len(milestones)},
        }
    )
    dataset_checks.append(
        {
            "name": "boundaries_and_interfaces_present",
            "passed": len(boundaries["risk_boundaries"]) >= 4 and len(boundaries["interfaces"]) >= 6,
            "details": {"boundary_count": len(boundaries["risk_boundaries"]), "interface_count": len(boundaries["interfaces"])},
        }
    )
    dataset_checks.append(
        {
            "name": "bottleneck_cost_org_outputs_present",
            "passed": len(bottlenecks) >= 3 and len(org_model["teams"]) >= 4 and cost_model["estimated_manual_review_hours"] > 0,
            "details": {"bottleneck_count": len(bottlenecks), "team_count": len(org_model["teams"])},
        }
    )
    dataset_checks.append(
        {
            "name": "dashboard_matches_metrics",
            "passed": dashboard["project_count"] == metrics["project_count"] and dashboard["bottleneck_count"] == metrics["bottleneck_count"],
            "details": dashboard,
        }
    )
    dataset_checks.append(
        {
            "name": "upstream_checks_aggregated",
            "passed": metrics["total_upstream_passed_checks"] == phase_inventory["total_passed_checks"],
            "details": {"metrics": metrics["total_upstream_passed_checks"], "inventory": phase_inventory["total_passed_checks"]},
        }
    )

    overall_passed = all(item["passed"] for item in command_checks + dataset_checks)
    results = {
        "overall_passed": overall_passed,
        "total_checks": len(command_checks) + len(dataset_checks),
        "passed_checks": sum(item["passed"] for item in command_checks + dataset_checks),
        "command_checks": command_checks,
        "dataset_checks": dataset_checks,
    }
    write_json(results, RESULTS_FILE)

    lines = [
        "# P10 Test Report",
        "",
        f"- Overall status: {'PASS' if overall_passed else 'FAIL'}",
        f"- Passed checks: {results['passed_checks']}/{results['total_checks']}",
        "",
        "## Command Checks",
        "",
    ]
    for item in command_checks:
        lines.append(f"- {item['name']}: {'PASS' if item['passed'] else 'FAIL'}")
        lines.append(f"  - Command: `{' '.join(item['command'])}`")
    lines.extend(["", "## Dataset Checks", ""])
    for item in dataset_checks:
        lines.append(f"- {item['name']}: {'PASS' if item['passed'] else 'FAIL'}")
        lines.append(f"  - Details: `{item['details']}`")

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print("✅ P10 测试完成。")
    print(results)


if __name__ == "__main__":
    main()
