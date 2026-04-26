from __future__ import annotations

from collections import Counter

from pipeline_utils import CONSOLE_DIR, PROCESSED_DIR, ensure_standard_dirs, write_json

SCOPE_FILE = PROCESSED_DIR / "platform_scope.json"
ARCH_FILE = PROCESSED_DIR / "architecture_spec.json"
API_FILE = PROCESSED_DIR / "api_catalog.json"
QUEUE_FILE = PROCESSED_DIR / "task_queues.json"
GOVERNANCE_FILE = PROCESSED_DIR / "governance_policy.json"
OPERATING_MODEL_FILE = PROCESSED_DIR / "operating_model.json"
UI_FILE = CONSOLE_DIR / "ui_panels.json"


def build_scope() -> dict:
    roles = {
        "platform_admin": [
            "manage_tenants",
            "manage_roles",
            "approve_exceptions",
            "trigger_rollbacks",
            "view_audit_logs",
        ],
        "data_engineer": [
            "register_dataset",
            "launch_ingestion",
            "publish_data_version",
            "view_lineage",
        ],
        "ml_engineer": [
            "launch_experiment",
            "compare_runs",
            "register_model_version",
            "request_rollbacks",
        ],
        "reviewer": [
            "approve_release",
            "review_quality_gate",
            "annotate_incident_review",
        ],
        "analyst": [
            "view_reports",
            "query_metadata",
            "monitor_sla_dashboard",
        ],
    }
    return {
        "platform_goal": "Unify ingestion, processing, evaluation, versioning, observability, and governance for enterprise data projects.",
        "tenants": ["foundation-data", "legal-ai", "finance-ai"],
        "projects": [
            {"project_id": "mini_c4_pipeline", "owner_team": "foundation-data", "project_type": "corpus_build"},
            {"project_id": "legal_sft_factory", "owner_team": "legal-ai", "project_type": "sft_data"},
            {"project_id": "finance_rag_ops", "owner_team": "finance-ai", "project_type": "rag_eval"},
        ],
        "roles": roles,
        "security_boundaries": [
            "dataset publish requires approval and audit logging",
            "rollback requires privileged role or approved exception",
            "project isolation prevents cross-tenant writes",
        ],
    }


def build_architecture() -> dict:
    return {
        "layers": [
            {
                "name": "scheduler_layer",
                "responsibilities": ["cron orchestration", "event triggers", "retry policy", "dependency scheduling"],
            },
            {
                "name": "metadata_layer",
                "responsibilities": ["dataset registry", "experiment tracking", "lineage graph", "approval records"],
            },
            {
                "name": "storage_layer",
                "responsibilities": ["raw zone", "processed zone", "artifact store", "feature snapshots"],
            },
            {
                "name": "service_layer",
                "responsibilities": ["ingestion service", "evaluation service", "release gate", "notification service"],
            },
        ],
        "runtime_components": {
            "task_queue": ["ingest_queue", "eval_queue", "release_queue", "backfill_queue"],
            "api_gateway": ["dataset API", "experiment API", "lineage API", "incident API"],
            "ui_modules": ["project overview", "version compare", "lineage graph", "ops dashboard", "audit center"],
        },
        "deployment_model": {
            "control_plane": "metadata, approvals, and queue coordination",
            "data_plane": "batch jobs, evaluators, and artifact processing",
        },
    }


def build_api_catalog() -> list[dict]:
    return [
        {"name": "POST /datasets/register", "purpose": "register a dataset version draft", "auth_role": "data_engineer"},
        {"name": "POST /experiments/launch", "purpose": "launch an experiment against a dataset version", "auth_role": "ml_engineer"},
        {"name": "GET /lineage/{asset_id}", "purpose": "retrieve upstream and downstream lineage", "auth_role": "analyst"},
        {"name": "POST /releases/promote", "purpose": "promote an approved artifact to production", "auth_role": "reviewer"},
        {"name": "POST /rollbacks/trigger", "purpose": "rollback a regressed model or dataset", "auth_role": "platform_admin"},
        {"name": "GET /ops/sla", "purpose": "read SLA and incident summaries", "auth_role": "analyst"},
    ]


def build_queues() -> list[dict]:
    return [
        {"queue_name": "ingest_queue", "priority": "high", "used_for": ["raw sync", "schema validation"]},
        {"queue_name": "eval_queue", "priority": "high", "used_for": ["quality scoring", "smoke tests", "regression checks"]},
        {"queue_name": "release_queue", "priority": "medium", "used_for": ["approval checks", "promotion", "rollback prepare"]},
        {"queue_name": "backfill_queue", "priority": "low", "used_for": ["historical recompute", "late audit enrichment"]},
    ]


def build_governance() -> dict:
    return {
        "team_interfaces": [
            {"producer": "platform_team", "consumer": "business_ai_team", "contract": "shared APIs, queue SLA, on-call support"},
            {"producer": "data_engineering", "consumer": "review_ops", "contract": "quality gates, release checklist, rollback playbook"},
        ],
        "standard_workflows": [
            "dataset draft -> quality evaluation -> reviewer approval -> release promotion",
            "experiment launch -> metric compare -> regression review -> register or rollback",
            "alert fired -> incident triage -> owner assignment -> postmortem and exception closure",
        ],
        "exception_process": {
            "required_fields": ["exception_id", "reason", "risk_assessment", "approver", "expiry_date"],
            "example_cases": ["temporary SLA breach waiver", "emergency rollback approval", "schema migration bypass"],
        },
    }


def build_ui_panels() -> list[dict]:
    return [
        {"panel_id": "project_overview", "widgets": ["active projects", "latest releases", "pending approvals"]},
        {"panel_id": "version_compare", "widgets": ["dataset diff", "metric compare", "rollback button"]},
        {"panel_id": "lineage_view", "widgets": ["graph canvas", "upstream assets", "downstream releases"]},
        {"panel_id": "ops_dashboard", "widgets": ["queue lag", "SLA status", "alerts", "incident MTTR"]},
        {"panel_id": "audit_center", "widgets": ["approval log", "actor timeline", "exception tracker"]},
    ]


def build_operating_model() -> dict:
    return {
        "raci_matrix": [
            {
                "workstream": "scope_and_intake",
                "platform_team": "A",
                "data_engineering": "R",
                "ml_team": "C",
                "review_ops": "I",
                "security_compliance": "C",
            },
            {
                "workstream": "release_gate_and_quality",
                "platform_team": "A",
                "data_engineering": "C",
                "ml_team": "R",
                "review_ops": "R",
                "security_compliance": "I",
            },
            {
                "workstream": "lineage_and_metadata_freshness",
                "platform_team": "A",
                "data_engineering": "R",
                "ml_team": "C",
                "review_ops": "I",
                "security_compliance": "I",
            },
            {
                "workstream": "incident_response_and_rollback",
                "platform_team": "A",
                "data_engineering": "C",
                "ml_team": "R",
                "review_ops": "C",
                "security_compliance": "I",
            },
            {
                "workstream": "exception_review_and_audit",
                "platform_team": "R",
                "data_engineering": "I",
                "ml_team": "I",
                "review_ops": "C",
                "security_compliance": "A",
            },
        ],
        "oncall_rotation": [
            {
                "tier": "L1_platform_ops",
                "coverage": "09:00-21:00 UTC",
                "owner": "platform_team",
                "focus": ["scheduler health", "queue lag", "console availability"],
            },
            {
                "tier": "L2_service_owner",
                "coverage": "follow-the-sun",
                "owner": "project owners",
                "focus": ["release gate failures", "quality regression", "lineage gaps"],
            },
            {
                "tier": "L3_platform_admin",
                "coverage": "major incidents only",
                "owner": "platform_admin",
                "focus": ["tenant isolation", "privileged rollback", "policy override"],
            },
        ],
        "operating_cadence": [
            {
                "cadence": "daily",
                "meeting": "ops standup",
                "inputs": ["open alerts", "queue backlog", "pending approvals"],
                "outputs": ["owner assignment", "same-day mitigation"],
            },
            {
                "cadence": "weekly",
                "meeting": "release and SLA review",
                "inputs": ["release gate pass rate", "rollback events", "SLA report"],
                "outputs": ["capacity tuning", "policy updates", "weekly summary"],
            },
            {
                "cadence": "monthly",
                "meeting": "governance board",
                "inputs": ["audit exceptions", "tenant roadmap", "cost trend"],
                "outputs": ["budget decision", "roadmap reprioritization", "control backlog"],
            },
        ],
        "escalation_policy": {
            "sev1": "page L1 immediately, engage L2 within 10 minutes, platform_admin approves rollback or freeze.",
            "sev2": "assign owner during business hours, resolve within the same shift, review in weekly ops.",
            "sev3": "record in backlog and audit summary, close through standard workflow.",
        },
    }


def main() -> None:
    ensure_standard_dirs()
    scope = build_scope()
    architecture = build_architecture()
    apis = build_api_catalog()
    queues = build_queues()
    governance = build_governance()
    ui_panels = build_ui_panels()
    operating_model = build_operating_model()

    summary = {
        "tenant_count": len(scope["tenants"]),
        "project_count": len(scope["projects"]),
        "role_count": len(scope["roles"]),
        "layer_count": len(architecture["layers"]),
        "api_count": len(apis),
        "queue_count": len(queues),
        "ui_panel_count": len(ui_panels),
        "raci_workstream_count": len(operating_model["raci_matrix"]),
        "oncall_tier_count": len(operating_model["oncall_rotation"]),
        "cadence_count": len(operating_model["operating_cadence"]),
        "project_type_distribution": dict(Counter(item["project_type"] for item in scope["projects"])),
    }

    write_json(scope, SCOPE_FILE)
    write_json(architecture, ARCH_FILE)
    write_json(apis, API_FILE)
    write_json(queues, QUEUE_FILE)
    write_json(governance, GOVERNANCE_FILE)
    write_json(operating_model, OPERATING_MODEL_FILE)
    write_json(ui_panels, UI_FILE)
    print("✅ P8 平台范围、架构与治理规格生成完成。")
    print(summary)


if __name__ == "__main__":
    main()
