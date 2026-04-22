from __future__ import annotations

import subprocess
from pathlib import Path

from pipeline_utils import CONSOLE_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_standard_dirs, load_json, load_jsonl, write_json

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
RESULTS_FILE = REPORTS_DIR / "p8_test_results.json"
REPORT_FILE = REPORTS_DIR / "p8_test_report.md"


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
    scope = load_json(PROCESSED_DIR / "platform_scope.json")
    architecture = load_json(PROCESSED_DIR / "architecture_spec.json")
    apis = load_json(PROCESSED_DIR / "api_catalog.json")
    queues = load_json(PROCESSED_DIR / "task_queues.json")
    governance = load_json(PROCESSED_DIR / "governance_policy.json")
    operating_model = load_json(PROCESSED_DIR / "operating_model.json")
    versions = load_jsonl(PROCESSED_DIR / "dataset_versions.jsonl")
    experiments = load_jsonl(PROCESSED_DIR / "experiment_runs.jsonl")
    lineage = load_json(PROCESSED_DIR / "lineage_graph.json")
    rollbacks = load_jsonl(PROCESSED_DIR / "rollback_events.jsonl")
    alerts = load_jsonl(PROCESSED_DIR / "alerts.jsonl")
    audit_log = load_jsonl(PROCESSED_DIR / "audit_log.jsonl")
    incidents = load_jsonl(PROCESSED_DIR / "incident_reviews.jsonl")
    sla_report = load_json(PROCESSED_DIR / "sla_report.json")
    ui_panels = load_json(CONSOLE_DIR / "ui_panels.json")

    py_files = sorted(str(path) for path in SRC_DIR.glob("*.py"))
    command_checks = [
        run_command(["python", "-m", "py_compile", *py_files], "py_compile"),
        run_command(["python", str(SRC_DIR / "evaluate_platform.py")], "evaluate_platform"),
    ]

    dataset_checks = []
    dataset_checks.append(
        {
            "name": "required_files_exist",
            "passed": all(
                path.exists()
                for path in [
                    PROCESSED_DIR / "platform_scope.json",
                    PROCESSED_DIR / "architecture_spec.json",
                    PROCESSED_DIR / "api_catalog.json",
                    PROCESSED_DIR / "task_queues.json",
                    PROCESSED_DIR / "governance_policy.json",
                    PROCESSED_DIR / "operating_model.json",
                    PROCESSED_DIR / "dataset_versions.jsonl",
                    PROCESSED_DIR / "experiment_runs.jsonl",
                    PROCESSED_DIR / "lineage_graph.json",
                    PROCESSED_DIR / "alerts.jsonl",
                    PROCESSED_DIR / "audit_log.jsonl",
                    PROCESSED_DIR / "incident_reviews.jsonl",
                    PROCESSED_DIR / "sla_report.json",
                    CONSOLE_DIR / "ui_panels.json",
                ]
            ),
            "details": {},
        }
    )
    dataset_checks.append(
        {
            "name": "role_and_permission_model_present",
            "passed": len(scope["roles"]) >= 5 and all(len(perms) >= 3 for perms in scope["roles"].values()),
            "details": {"role_names": sorted(scope["roles"])},
        }
    )
    dataset_checks.append(
        {
            "name": "architecture_layers_complete",
            "passed": {"scheduler_layer", "metadata_layer", "storage_layer", "service_layer"} == {layer["name"] for layer in architecture["layers"]},
            "details": {"layer_names": [layer["name"] for layer in architecture["layers"]]},
        }
    )
    dataset_checks.append(
        {
            "name": "api_queue_ui_present",
            "passed": len(apis) >= 5 and len(queues) >= 4 and len(ui_panels) >= 5,
            "details": {"api_count": len(apis), "queue_count": len(queues), "ui_panel_count": len(ui_panels)},
        }
    )
    version_ids = {item["version_id"] for item in versions}
    dataset_checks.append(
        {
            "name": "version_lineage_links_valid",
            "passed": all(item["parent_version"] in version_ids or item["parent_version"] is None for item in versions),
            "details": {"version_count": len(versions)},
        }
    )
    experiment_dataset_refs_ok = all(item["dataset_version"] in version_ids for item in experiments)
    dataset_checks.append(
        {
            "name": "experiments_reference_versions",
            "passed": experiment_dataset_refs_ok,
            "details": {"experiment_count": len(experiments)},
        }
    )
    rolled_back_assets = {item["trigger_asset"] for item in rollbacks}
    regressed_assets = {item["artifact_id"] for item in experiments if item["status"] == "regressed"}
    dataset_checks.append(
        {
            "name": "regressions_have_rollback",
            "passed": regressed_assets.issubset(rolled_back_assets),
            "details": {"regressed_assets": sorted(regressed_assets), "rolled_back_assets": sorted(rolled_back_assets)},
        }
    )
    dataset_checks.append(
        {
            "name": "alerts_audit_incidents_present",
            "passed": len(alerts) >= 3 and len(audit_log) >= 5 and len(incidents) >= 2,
            "details": {"alert_count": len(alerts), "audit_count": len(audit_log), "incident_count": len(incidents)},
        }
    )
    dataset_checks.append(
        {
            "name": "sla_all_met",
            "passed": all(value == "met" for value in sla_report["status"].values()),
            "details": sla_report["status"],
        }
    )
    lineage_ids = {node["id"] for node in lineage["nodes"]}
    dataset_checks.append(
        {
            "name": "lineage_edges_resolve",
            "passed": all(edge["source"] in lineage_ids and edge["target"] in lineage_ids for edge in lineage["edges"]),
            "details": {"node_count": len(lineage["nodes"]), "edge_count": len(lineage["edges"])},
        }
    )
    dataset_checks.append(
        {
            "name": "governance_exception_process_present",
            "passed": "exception_process" in governance and len(governance["standard_workflows"]) >= 3,
            "details": {"workflow_count": len(governance["standard_workflows"])},
        }
    )
    dataset_checks.append(
        {
            "name": "operating_model_raci_complete",
            "passed": len(operating_model["raci_matrix"]) >= 5
            and all(
                {"workstream", "platform_team", "data_engineering", "ml_team", "review_ops", "security_compliance"}.issubset(item)
                for item in operating_model["raci_matrix"]
            ),
            "details": {"raci_workstream_count": len(operating_model["raci_matrix"])},
        }
    )
    dataset_checks.append(
        {
            "name": "operating_model_oncall_and_cadence_present",
            "passed": len(operating_model["oncall_rotation"]) >= 3 and len(operating_model["operating_cadence"]) >= 3,
            "details": {
                "oncall_tier_count": len(operating_model["oncall_rotation"]),
                "cadence_count": len(operating_model["operating_cadence"]),
            },
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
        "# P8 Test Report",
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
    print("✅ P8 测试完成。")
    print(results)


if __name__ == "__main__":
    main()
