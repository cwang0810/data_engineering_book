from __future__ import annotations

import re
import subprocess
from pathlib import Path

from pipeline_utils import PROCESSED_DIR, REPORTS_DIR, ensure_standard_dirs, load_json, load_jsonl, write_json

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
RESULTS_FILE = REPORTS_DIR / "p9_test_results.json"
REPORT_FILE = REPORTS_DIR / "p9_test_report.md"

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
BANK_RE = re.compile(r"\b\d{10,12}\b")
PATIENT_RE = re.compile(r"\bPT-\d{4,6}\b")


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


def has_direct_pii(text: str) -> bool:
    return any(regex.search(text) for regex in [EMAIL_RE, PHONE_RE, SSN_RE, BANK_RE, PATIENT_RE])


def main() -> None:
    ensure_standard_dirs()
    scope = load_json(PROCESSED_DIR / "compliance_scope.json")
    classification = load_json(PROCESSED_DIR / "classification_policy.json")
    access = load_json(PROCESSED_DIR / "access_policy.json")
    tech = load_json(PROCESSED_DIR / "privacy_tech_options.json")
    raw_records = load_jsonl(PROCESSED_DIR / "raw_sensitive_records.jsonl")
    classified = load_jsonl(PROCESSED_DIR / "classified_records.jsonl")
    redacted = load_jsonl(PROCESSED_DIR / "redacted_records.jsonl")
    quarantined = load_jsonl(PROCESSED_DIR / "quarantine_records.jsonl")
    alerts = load_jsonl(PROCESSED_DIR / "access_alerts.jsonl")
    audit_log = load_jsonl(PROCESSED_DIR / "audit_log.jsonl")
    preflight = load_json(PROCESSED_DIR / "preflight_checklist.json")
    incident = load_json(PROCESSED_DIR / "incident_simulation.json")
    postmortem = load_json(PROCESSED_DIR / "postmortem_report.json")

    py_files = sorted(str(path) for path in SRC_DIR.glob("*.py"))
    command_checks = [
        run_command(["python", "-m", "py_compile", *py_files], "py_compile"),
        run_command(["python", str(SRC_DIR / "evaluate_privacy_pipeline.py")], "evaluate_privacy_pipeline"),
    ]

    dataset_checks = []
    dataset_checks.append(
        {
            "name": "required_files_exist",
            "passed": all(
                path.exists()
                for path in [
                    PROCESSED_DIR / "compliance_scope.json",
                    PROCESSED_DIR / "classification_policy.json",
                    PROCESSED_DIR / "access_policy.json",
                    PROCESSED_DIR / "privacy_tech_options.json",
                    PROCESSED_DIR / "raw_sensitive_records.jsonl",
                    PROCESSED_DIR / "classified_records.jsonl",
                    PROCESSED_DIR / "redacted_records.jsonl",
                    PROCESSED_DIR / "quarantine_records.jsonl",
                    PROCESSED_DIR / "audit_log.jsonl",
                    PROCESSED_DIR / "access_alerts.jsonl",
                    PROCESSED_DIR / "preflight_checklist.json",
                    PROCESSED_DIR / "incident_simulation.json",
                    PROCESSED_DIR / "postmortem_report.json",
                ]
            ),
            "details": {},
        }
    )
    dataset_checks.append(
        {
            "name": "role_and_zone_model_present",
            "passed": len(access["roles"]) >= 5 and len(access["storage_zones"]) >= 4,
            "details": {"role_count": len(access["roles"]), "storage_zone_count": len(access["storage_zones"])},
        }
    )
    dataset_checks.append(
        {
            "name": "all_records_classified",
            "passed": len(raw_records) == len(classified) and all("sensitivity_level" in item for item in classified),
            "details": {"raw_record_count": len(raw_records), "classified_record_count": len(classified)},
        }
    )
    dataset_checks.append(
        {
            "name": "restricted_records_quarantined",
            "passed": sum(item["sensitivity_level"] == "restricted" for item in classified) == len(quarantined),
            "details": {"restricted_count": sum(item["sensitivity_level"] == "restricted" for item in classified), "quarantine_count": len(quarantined)},
        }
    )
    dataset_checks.append(
        {
            "name": "redacted_records_remove_direct_pii",
            "passed": all(not has_direct_pii(item["payload"]) for item in redacted),
            "details": {"redacted_record_count": len(redacted)},
        }
    )
    dataset_checks.append(
        {
            "name": "pii_detection_rules_present",
            "passed": {"email", "phone", "ssn", "bank_account", "patient_id"} == {item["pattern_name"] for item in classification["pii_rules"]},
            "details": {"pii_rule_names": [item["pattern_name"] for item in classification["pii_rules"]]},
        }
    )
    dataset_checks.append(
        {
            "name": "alerts_and_audit_present",
            "passed": len(alerts) >= 2 and len(audit_log) >= 5,
            "details": {"alert_count": len(alerts), "audit_event_count": len(audit_log)},
        }
    )
    dataset_checks.append(
        {
            "name": "alerts_resolved",
            "passed": all(item["status"] == "resolved" for item in alerts),
            "details": {"resolved_alerts": sum(item["status"] == "resolved" for item in alerts)},
        }
    )
    dataset_checks.append(
        {
            "name": "privacy_tech_options_complete",
            "passed": {"differential_privacy", "tee", "fhe", "tokenization"} == {item["technology"] for item in tech},
            "details": {"technology_names": [item["technology"] for item in tech]},
        }
    )
    dataset_checks.append(
        {
            "name": "preflight_all_passed",
            "passed": preflight["overall_passed"],
            "details": {"passed_checks": preflight["passed_checks"], "total_checks": preflight["total_checks"]},
        }
    )
    dataset_checks.append(
        {
            "name": "incident_and_postmortem_present",
            "passed": incident["response_minutes"] <= 30 and len(postmortem["follow_ups"]) >= 3,
            "details": {"response_minutes": incident["response_minutes"], "follow_up_count": len(postmortem["follow_ups"])},
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
        "# P9 Test Report",
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
    print("✅ P9 测试完成。")
    print(results)


if __name__ == "__main__":
    main()
