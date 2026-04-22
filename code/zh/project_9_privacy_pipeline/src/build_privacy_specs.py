from __future__ import annotations

from collections import Counter

from pipeline_utils import PROCESSED_DIR, ensure_standard_dirs, write_json

SCOPE_FILE = PROCESSED_DIR / "compliance_scope.json"
CLASSIFICATION_FILE = PROCESSED_DIR / "classification_policy.json"
ACCESS_FILE = PROCESSED_DIR / "access_policy.json"
TECH_FILE = PROCESSED_DIR / "privacy_tech_options.json"


def build_scope() -> dict:
    return {
        "pipeline_goal": "Build a reproducible privacy-preserving data processing pipeline for highly sensitive records.",
        "example_domains": ["healthcare_support", "payroll_hr", "financial_kyc"],
        "compliance_targets": ["least_privilege", "auditability", "de-identification before analytics", "incident response readiness"],
        "risk_goals": [
            "prevent direct PII leakage to analytics consumers",
            "separate raw storage from redacted processing zones",
            "log suspicious access and export attempts",
        ],
    }


def build_classification_policy() -> dict:
    return {
        "sensitivity_levels": [
            {"level": "restricted", "description": "direct PII, health identifiers, payroll details, bank data"},
            {"level": "confidential", "description": "internal case details and support notes"},
            {"level": "internal", "description": "aggregate metrics and sanitized analytics outputs"},
        ],
        "source_types": [
            {"source_type": "support_ticket", "default_level": "confidential"},
            {"source_type": "employee_payroll", "default_level": "restricted"},
            {"source_type": "kyc_form", "default_level": "restricted"},
            {"source_type": "analytics_export", "default_level": "internal"},
        ],
        "pii_rules": [
            {"pattern_name": "email", "action": "tokenize"},
            {"pattern_name": "phone", "action": "mask"},
            {"pattern_name": "ssn", "action": "remove"},
            {"pattern_name": "bank_account", "action": "tokenize"},
            {"pattern_name": "patient_id", "action": "tokenize"},
        ],
    }


def build_access_policy() -> dict:
    roles = {
        "privacy_admin": ["read_raw_zone", "approve_exports", "review_alerts", "trigger_incident_lockdown"],
        "data_processor": ["write_redacted_zone", "run_deid_jobs", "view_classification_reports"],
        "analyst": ["read_redacted_zone", "view_aggregates"],
        "auditor": ["read_audit_log", "read_alerts", "review_exception_cases"],
        "incident_responder": ["read_alerts", "lock_accounts", "quarantine_exports", "annotate_postmortem"],
    }
    return {
        "roles": roles,
        "boundary_rules": [
            "raw_zone is accessible only by privacy_admin",
            "redacted_zone is accessible by data_processor and analyst",
            "export requires privacy_admin approval and audit record",
            "suspicious cross-role access triggers alert and temporary quarantine",
        ],
        "storage_zones": [
            {"zone_name": "raw_zone", "encryption": "required", "network_scope": "isolated"},
            {"zone_name": "quarantine_zone", "encryption": "required", "network_scope": "isolated"},
            {"zone_name": "redacted_zone", "encryption": "required", "network_scope": "analytics"},
            {"zone_name": "audit_zone", "encryption": "required", "network_scope": "security"},
        ],
    }


def build_tech_options() -> list[dict]:
    return [
        {
            "technology": "differential_privacy",
            "fit": "aggregate reporting and analytics release",
            "benefit": "reduces re-identification risk for published statistics",
            "cost_tradeoff": "slight metric noise and tuning overhead",
        },
        {
            "technology": "tee",
            "fit": "isolated processing of raw restricted records",
            "benefit": "stronger runtime isolation for sensitive compute",
            "cost_tradeoff": "higher infra complexity and hardware dependency",
        },
        {
            "technology": "fhe",
            "fit": "future encrypted computation on highest-risk fields",
            "benefit": "compute on encrypted values without plaintext exposure",
            "cost_tradeoff": "significant latency and engineering cost",
        },
        {
            "technology": "tokenization",
            "fit": "operational de-identification before analytics",
            "benefit": "practical and cheap for replacing direct identifiers",
            "cost_tradeoff": "still needs secure token vault management",
        },
    ]


def main() -> None:
    ensure_standard_dirs()
    scope = build_scope()
    classification = build_classification_policy()
    access = build_access_policy()
    tech = build_tech_options()
    summary = {
        "domain_count": len(scope["example_domains"]),
        "compliance_target_count": len(scope["compliance_targets"]),
        "role_count": len(access["roles"]),
        "storage_zone_count": len(access["storage_zones"]),
        "technology_count": len(tech),
        "source_type_distribution": dict(Counter(item["default_level"] for item in classification["source_types"])),
    }
    write_json(scope, SCOPE_FILE)
    write_json(classification, CLASSIFICATION_FILE)
    write_json(access, ACCESS_FILE)
    write_json(tech, TECH_FILE)
    print("✅ P9 合规、分类、权限与隐私技术规格生成完成。")
    print(summary)


if __name__ == "__main__":
    main()
