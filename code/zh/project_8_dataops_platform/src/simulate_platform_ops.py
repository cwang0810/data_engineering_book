from __future__ import annotations

from collections import Counter

from pipeline_utils import PROCESSED_DIR, ensure_standard_dirs, write_json, write_jsonl

VERSIONS_FILE = PROCESSED_DIR / "dataset_versions.jsonl"
EXPERIMENTS_FILE = PROCESSED_DIR / "experiment_runs.jsonl"
LINEAGE_FILE = PROCESSED_DIR / "lineage_graph.json"
ROLLBACK_FILE = PROCESSED_DIR / "rollback_events.jsonl"
ALERTS_FILE = PROCESSED_DIR / "alerts.jsonl"
AUDIT_FILE = PROCESSED_DIR / "audit_log.jsonl"
INCIDENTS_FILE = PROCESSED_DIR / "incident_reviews.jsonl"
SLA_FILE = PROCESSED_DIR / "sla_report.json"
SUMMARY_FILE = PROCESSED_DIR / "ops_summary.json"


def build_dataset_versions() -> list[dict]:
    return [
        {
            "version_id": "mini_c4_v1",
            "project_id": "mini_c4_pipeline",
            "parent_version": None,
            "status": "released",
            "quality_score": 0.81,
            "record_count": 526,
            "owner": "foundation-data",
        },
        {
            "version_id": "mini_c4_v2",
            "project_id": "mini_c4_pipeline",
            "parent_version": "mini_c4_v1",
            "status": "released",
            "quality_score": 0.86,
            "record_count": 610,
            "owner": "foundation-data",
        },
        {
            "version_id": "legal_sft_v1",
            "project_id": "legal_sft_factory",
            "parent_version": None,
            "status": "released",
            "quality_score": 0.84,
            "record_count": 7737,
            "owner": "legal-ai",
        },
        {
            "version_id": "legal_sft_v2",
            "project_id": "legal_sft_factory",
            "parent_version": "legal_sft_v1",
            "status": "candidate",
            "quality_score": 0.82,
            "record_count": 8120,
            "owner": "legal-ai",
        },
        {
            "version_id": "finance_rag_v1",
            "project_id": "finance_rag_ops",
            "parent_version": None,
            "status": "released",
            "quality_score": 0.88,
            "record_count": 1341,
            "owner": "finance-ai",
        },
        {
            "version_id": "finance_rag_v2",
            "project_id": "finance_rag_ops",
            "parent_version": "finance_rag_v1",
            "status": "released",
            "quality_score": 0.91,
            "record_count": 1488,
            "owner": "finance-ai",
        },
    ]


def build_experiments() -> list[dict]:
    return [
        {
            "run_id": "exp_mini_c4_smoke_001",
            "project_id": "mini_c4_pipeline",
            "dataset_version": "mini_c4_v1",
            "artifact_id": "model_mini_001",
            "status": "completed",
            "metric_primary": 0.62,
            "latency_ms": 105,
            "release_decision": "accepted",
        },
        {
            "run_id": "exp_mini_c4_smoke_002",
            "project_id": "mini_c4_pipeline",
            "dataset_version": "mini_c4_v2",
            "artifact_id": "model_mini_002",
            "status": "completed",
            "metric_primary": 0.68,
            "latency_ms": 99,
            "release_decision": "accepted",
        },
        {
            "run_id": "exp_legal_lora_001",
            "project_id": "legal_sft_factory",
            "dataset_version": "legal_sft_v1",
            "artifact_id": "model_legal_001",
            "status": "completed",
            "metric_primary": 0.77,
            "latency_ms": 121,
            "release_decision": "accepted",
        },
        {
            "run_id": "exp_legal_lora_002",
            "project_id": "legal_sft_factory",
            "dataset_version": "legal_sft_v2",
            "artifact_id": "model_legal_002",
            "status": "regressed",
            "metric_primary": 0.72,
            "latency_ms": 145,
            "release_decision": "rolled_back",
        },
        {
            "run_id": "exp_finance_retriever_001",
            "project_id": "finance_rag_ops",
            "dataset_version": "finance_rag_v1",
            "artifact_id": "retriever_fin_001",
            "status": "completed",
            "metric_primary": 0.89,
            "latency_ms": 88,
            "release_decision": "accepted",
        },
        {
            "run_id": "exp_finance_retriever_002",
            "project_id": "finance_rag_ops",
            "dataset_version": "finance_rag_v2",
            "artifact_id": "retriever_fin_002",
            "status": "completed",
            "metric_primary": 0.93,
            "latency_ms": 81,
            "release_decision": "accepted",
        },
        {
            "run_id": "exp_finance_generator_003",
            "project_id": "finance_rag_ops",
            "dataset_version": "finance_rag_v2",
            "artifact_id": "generator_fin_003",
            "status": "failed",
            "metric_primary": 0.0,
            "latency_ms": 220,
            "release_decision": "retry_required",
        },
    ]


def build_rollbacks() -> list[dict]:
    return [
        {
            "rollback_id": "rb_legal_001",
            "trigger_asset": "model_legal_002",
            "fallback_asset": "model_legal_001",
            "reason": "faithfulness regression and latency increase",
            "approved_by": "platform_admin",
            "status": "completed",
        }
    ]


def build_alerts() -> list[dict]:
    return [
        {"alert_id": "alert_001", "severity": "high", "source": "release_gate", "message": "legal_sft_v2 quality regression", "status": "resolved"},
        {"alert_id": "alert_002", "severity": "medium", "source": "scheduler", "message": "finance generator queue latency spike", "status": "resolved"},
        {"alert_id": "alert_003", "severity": "low", "source": "audit", "message": "one approval note was missing reviewer comment", "status": "resolved"},
    ]


def build_audit_log() -> list[dict]:
    return [
        {"event_id": "audit_001", "actor": "data_engineer", "action": "publish_dataset", "target": "mini_c4_v2"},
        {"event_id": "audit_002", "actor": "ml_engineer", "action": "launch_experiment", "target": "exp_legal_lora_002"},
        {"event_id": "audit_003", "actor": "reviewer", "action": "reject_release", "target": "model_legal_002"},
        {"event_id": "audit_004", "actor": "platform_admin", "action": "trigger_rollback", "target": "rb_legal_001"},
        {"event_id": "audit_005", "actor": "analyst", "action": "view_sla_dashboard", "target": "finance_rag_ops"},
    ]


def build_incidents() -> list[dict]:
    return [
        {
            "incident_id": "inc_001",
            "project_id": "legal_sft_factory",
            "root_cause": "dataset refresh lowered answer faithfulness on validation",
            "mttr_minutes": 42,
            "follow_up": "tighten release gate on citation consistency",
        },
        {
            "incident_id": "inc_002",
            "project_id": "finance_rag_ops",
            "root_cause": "generator retry loop increased queue latency",
            "mttr_minutes": 31,
            "follow_up": "separate generator retries into backfill queue",
        },
    ]


def build_sla_report() -> dict:
    return {
        "dataset_publish_sla_hours": 6,
        "incident_response_sla_minutes": 30,
        "lineage_freshness_sla_minutes": 15,
        "measured": {
            "dataset_publish_p95_hours": 3.2,
            "incident_response_p95_minutes": 28,
            "lineage_freshness_p95_minutes": 9,
        },
        "status": {
            "dataset_publish": "met",
            "incident_response": "met",
            "lineage_freshness": "met",
        },
    }


def build_lineage(versions: list[dict], experiments: list[dict], rollbacks: list[dict]) -> dict:
    nodes = []
    edges = []
    for version in versions:
        nodes.append({"id": version["version_id"], "type": "dataset_version", "project_id": version["project_id"]})
        if version["parent_version"]:
            edges.append({"source": version["parent_version"], "target": version["version_id"], "relation": "derived_from"})
    for experiment in experiments:
        nodes.append({"id": experiment["run_id"], "type": "experiment_run", "project_id": experiment["project_id"]})
        nodes.append({"id": experiment["artifact_id"], "type": "artifact", "project_id": experiment["project_id"]})
        edges.append({"source": experiment["dataset_version"], "target": experiment["run_id"], "relation": "used_by"})
        edges.append({"source": experiment["run_id"], "target": experiment["artifact_id"], "relation": "produced"})
    for rollback in rollbacks:
        nodes.append({"id": rollback["rollback_id"], "type": "rollback_event", "project_id": "legal_sft_factory"})
        edges.append({"source": rollback["trigger_asset"], "target": rollback["rollback_id"], "relation": "rolled_back_by"})
        edges.append({"source": rollback["rollback_id"], "target": rollback["fallback_asset"], "relation": "restored"})
    return {"nodes": nodes, "edges": edges}


def main() -> None:
    ensure_standard_dirs()
    versions = build_dataset_versions()
    experiments = build_experiments()
    rollbacks = build_rollbacks()
    alerts = build_alerts()
    audit_log = build_audit_log()
    incidents = build_incidents()
    sla_report = build_sla_report()
    lineage = build_lineage(versions, experiments, rollbacks)

    summary = {
        "dataset_version_count": len(versions),
        "experiment_count": len(experiments),
        "rollback_count": len(rollbacks),
        "alert_count": len(alerts),
        "audit_event_count": len(audit_log),
        "incident_count": len(incidents),
        "released_version_count": sum(item["status"] == "released" for item in versions),
        "experiment_status_distribution": dict(Counter(item["status"] for item in experiments)),
    }

    write_jsonl(versions, VERSIONS_FILE)
    write_jsonl(experiments, EXPERIMENTS_FILE)
    write_json(lineage, LINEAGE_FILE)
    write_jsonl(rollbacks, ROLLBACK_FILE)
    write_jsonl(alerts, ALERTS_FILE)
    write_jsonl(audit_log, AUDIT_FILE)
    write_jsonl(incidents, INCIDENTS_FILE)
    write_json(sla_report, SLA_FILE)
    write_json(summary, SUMMARY_FILE)
    print("✅ P8 版本、实验、监控与治理链路模拟完成。")
    print(summary)


if __name__ == "__main__":
    main()
