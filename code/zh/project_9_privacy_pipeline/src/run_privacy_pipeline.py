from __future__ import annotations

import re
from collections import Counter

from pipeline_utils import PROCESSED_DIR, ensure_standard_dirs, hash_token, load_json, write_json, write_jsonl

RAW_FILE = PROCESSED_DIR / "raw_sensitive_records.jsonl"
CLASSIFIED_FILE = PROCESSED_DIR / "classified_records.jsonl"
REDACTED_FILE = PROCESSED_DIR / "redacted_records.jsonl"
QUARANTINE_FILE = PROCESSED_DIR / "quarantine_records.jsonl"
AUDIT_FILE = PROCESSED_DIR / "audit_log.jsonl"
ALERTS_FILE = PROCESSED_DIR / "access_alerts.jsonl"
ISOLATION_FILE = PROCESSED_DIR / "isolation_plan.json"
SUMMARY_FILE = PROCESSED_DIR / "privacy_pipeline_summary.json"

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
BANK_RE = re.compile(r"\b\d{10,12}\b")
PATIENT_RE = re.compile(r"\bPT-\d{4,6}\b")


def build_raw_records() -> list[dict]:
    return [
        {
            "record_id": "rec_001",
            "source_type": "support_ticket",
            "domain": "healthcare_support",
            "owner_team": "care_ops",
            "payload": "Patient John Lee, patient id PT-483920, email john.lee@example.com, phone 415-555-2198 asked about claim denial.",
        },
        {
            "record_id": "rec_002",
            "source_type": "employee_payroll",
            "domain": "payroll_hr",
            "owner_team": "hr_ops",
            "payload": "Employee Marta Chen SSN 342-19-8842 salary adjustment note for payroll cycle 2026-04.",
        },
        {
            "record_id": "rec_003",
            "source_type": "kyc_form",
            "domain": "financial_kyc",
            "owner_team": "fin_ops",
            "payload": "KYC form for beta@corp.test with bank account 998877665544 and risk review pending.",
        },
        {
            "record_id": "rec_004",
            "source_type": "support_ticket",
            "domain": "healthcare_support",
            "owner_team": "care_ops",
            "payload": "Follow-up ticket for patient PT-555888, callback 212-555-0100, no diagnosis shared.",
        },
        {
            "record_id": "rec_005",
            "source_type": "analytics_export",
            "domain": "financial_kyc",
            "owner_team": "analytics",
            "payload": "Weekly aggregate: 18 forms reviewed, 2 pending, 0 escalated.",
        },
        {
            "record_id": "rec_006",
            "source_type": "employee_payroll",
            "domain": "payroll_hr",
            "owner_team": "hr_ops",
            "payload": "Payroll support case for sam.green@example.com direct deposit account 123456789012 requires verification.",
        },
        {
            "record_id": "rec_007",
            "source_type": "kyc_form",
            "domain": "financial_kyc",
            "owner_team": "fin_ops",
            "payload": "Applicant id review for arya@startup.test, analyst requested additional address proof.",
        },
        {
            "record_id": "rec_008",
            "source_type": "support_ticket",
            "domain": "healthcare_support",
            "owner_team": "care_ops",
            "payload": "Ticket notes mention guardian contact at 650-555-8821 and email family.contact@test.org.",
        },
    ]


def detect_pii(text: str) -> list[dict]:
    detections: list[dict] = []
    for pattern_name, regex in [
        ("email", EMAIL_RE),
        ("phone", PHONE_RE),
        ("ssn", SSN_RE),
        ("bank_account", BANK_RE),
        ("patient_id", PATIENT_RE),
    ]:
        for match in regex.finditer(text):
            detections.append({"pattern_name": pattern_name, "match": match.group(0)})
    return detections


def redact_payload(text: str, detections: list[dict]) -> str:
    redacted = text
    for detection in detections:
        match = detection["match"]
        if detection["pattern_name"] in {"email", "bank_account", "patient_id"}:
            replacement = hash_token(match)
        elif detection["pattern_name"] == "phone":
            replacement = "***-***-" + match[-4:]
        else:
            replacement = "[REMOVED_SSN]"
        redacted = redacted.replace(match, replacement)
    return redacted


def classify_record(record: dict, classification_policy: dict) -> dict:
    source_type_map = {item["source_type"]: item["default_level"] for item in classification_policy["source_types"]}
    detections = detect_pii(record["payload"])
    sensitivity = source_type_map.get(record["source_type"], "internal")
    if detections:
        sensitivity = "restricted"
    return {
        **record,
        "sensitivity_level": sensitivity,
        "pii_detections": detections,
        "requires_quarantine": sensitivity == "restricted",
    }


def build_audit_log() -> list[dict]:
    return [
        {"event_id": "audit_001", "actor": "privacy_admin", "action": "ingest_raw_batch", "target": "raw_zone"},
        {"event_id": "audit_002", "actor": "data_processor", "action": "run_deid_job", "target": "redacted_zone"},
        {"event_id": "audit_003", "actor": "analyst", "action": "read_redacted_export", "target": "redacted_zone"},
        {"event_id": "audit_004", "actor": "analyst", "action": "attempt_read_raw_zone", "target": "raw_zone"},
        {"event_id": "audit_005", "actor": "incident_responder", "action": "quarantine_export", "target": "quarantine_zone"},
    ]


def build_alerts() -> list[dict]:
    return [
        {
            "alert_id": "alert_priv_001",
            "severity": "high",
            "actor": "analyst",
            "reason": "unauthorized raw zone access attempt",
            "status": "resolved",
        },
        {
            "alert_id": "alert_priv_002",
            "severity": "medium",
            "actor": "data_processor",
            "reason": "restricted export requested without approval",
            "status": "resolved",
        },
    ]


def build_isolation_plan() -> dict:
    return {
        "zones": [
            {"zone_name": "raw_zone", "store": "encrypted object storage", "access": ["privacy_admin"]},
            {"zone_name": "quarantine_zone", "store": "isolated secure bucket", "access": ["privacy_admin", "incident_responder"]},
            {"zone_name": "redacted_zone", "store": "analytics warehouse", "access": ["data_processor", "analyst"]},
            {"zone_name": "audit_zone", "store": "security log store", "access": ["auditor", "privacy_admin"]},
        ],
        "deid_flow": [
            "ingest raw restricted records",
            "classify and detect PII",
            "write restricted originals to raw_zone",
            "redact identifiers and emit sanitized records to redacted_zone",
            "quarantine flagged export attempts and emit audit alerts",
        ],
    }


def main() -> None:
    ensure_standard_dirs()
    classification_policy = load_json(PROCESSED_DIR / "classification_policy.json")
    raw_records = build_raw_records()
    classified = [classify_record(record, classification_policy) for record in raw_records]
    redacted = []
    quarantined = []
    for record in classified:
        redacted_record = dict(record)
        redacted_record["payload"] = redact_payload(record["payload"], record["pii_detections"])
        redacted.append(redacted_record)
        if record["requires_quarantine"]:
            quarantined.append({"record_id": record["record_id"], "reason": "restricted_source", "owner_team": record["owner_team"]})

    audit_log = build_audit_log()
    alerts = build_alerts()
    isolation_plan = build_isolation_plan()
    summary = {
        "raw_record_count": len(raw_records),
        "classified_record_count": len(classified),
        "restricted_record_count": sum(item["sensitivity_level"] == "restricted" for item in classified),
        "redacted_record_count": len(redacted),
        "quarantine_count": len(quarantined),
        "pii_detection_distribution": dict(Counter(detection["pattern_name"] for item in classified for detection in item["pii_detections"])),
        "alert_count": len(alerts),
        "audit_event_count": len(audit_log),
    }

    write_jsonl(raw_records, RAW_FILE)
    write_jsonl(classified, CLASSIFIED_FILE)
    write_jsonl(redacted, REDACTED_FILE)
    write_jsonl(quarantined, QUARANTINE_FILE)
    write_jsonl(audit_log, AUDIT_FILE)
    write_jsonl(alerts, ALERTS_FILE)
    write_json(isolation_plan, ISOLATION_FILE)
    write_json(summary, SUMMARY_FILE)
    print("✅ P9 脱敏、隔离与安全处理流水线完成。")
    print(summary)


if __name__ == "__main__":
    main()
