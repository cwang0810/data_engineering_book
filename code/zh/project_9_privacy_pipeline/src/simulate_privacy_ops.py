from __future__ import annotations

from pipeline_utils import PROCESSED_DIR, ensure_standard_dirs, load_json, load_jsonl, write_json

PREFLIGHT_FILE = PROCESSED_DIR / "preflight_checklist.json"
INCIDENT_FILE = PROCESSED_DIR / "incident_simulation.json"
POSTMORTEM_FILE = PROCESSED_DIR / "postmortem_report.json"
OPS_SUMMARY_FILE = PROCESSED_DIR / "ops_summary.json"


def main() -> None:
    ensure_standard_dirs()
    classified = load_jsonl(PROCESSED_DIR / "classified_records.jsonl")
    alerts = load_jsonl(PROCESSED_DIR / "access_alerts.jsonl")
    access_policy = load_json(PROCESSED_DIR / "access_policy.json")
    tech_options = load_json(PROCESSED_DIR / "privacy_tech_options.json")

    preflight = {
        "checks": [
            {"name": "all records classified", "passed": len(classified) > 0 and all("sensitivity_level" in item for item in classified)},
            {"name": "restricted records isolated", "passed": all(item["requires_quarantine"] == (item["sensitivity_level"] == "restricted") for item in classified)},
            {"name": "alerts wired", "passed": len(alerts) >= 2},
            {"name": "role model present", "passed": len(access_policy["roles"]) >= 5},
            {"name": "privacy tech options documented", "passed": len(tech_options) >= 4},
        ]
    }
    preflight["passed_checks"] = sum(item["passed"] for item in preflight["checks"])
    preflight["total_checks"] = len(preflight["checks"])
    preflight["overall_passed"] = preflight["passed_checks"] == preflight["total_checks"]

    incident = {
        "incident_id": "privacy_inc_001",
        "scenario": "analyst attempted to export restricted raw records without approval",
        "detection": "access_alerts.jsonl high severity alert",
        "containment": [
            "quarantine the export request",
            "lock the analyst session",
            "require privacy_admin review",
        ],
        "outcome": "resolved with no confirmed external data leak",
        "response_minutes": 24,
    }

    postmortem = {
        "incident_id": incident["incident_id"],
        "root_cause": "role boundary was respected, but analyst attempted an out-of-policy access path",
        "what_worked": [
            "alert fired immediately",
            "quarantine workflow prevented raw export",
            "audit log preserved actor timeline",
        ],
        "follow_ups": [
            "add stronger export pre-check banner for analysts",
            "require dual approval for restricted export exceptions",
            "add DP gate for aggregate release workflows",
        ],
    }

    ops_summary = {
        "preflight_pass_rate": round(preflight["passed_checks"] / max(1, preflight["total_checks"]), 4),
        "incident_response_minutes": incident["response_minutes"],
        "follow_up_count": len(postmortem["follow_ups"]),
    }

    write_json(preflight, PREFLIGHT_FILE)
    write_json(incident, INCIDENT_FILE)
    write_json(postmortem, POSTMORTEM_FILE)
    write_json(ops_summary, OPS_SUMMARY_FILE)
    print("✅ P9 上线检查与事故复盘模拟完成。")
    print(ops_summary)


if __name__ == "__main__":
    main()
