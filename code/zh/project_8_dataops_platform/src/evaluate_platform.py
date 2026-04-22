from __future__ import annotations

from collections import Counter

from pipeline_utils import PROCESSED_DIR, REPORTS_DIR, ensure_standard_dirs, load_json, load_jsonl, write_json

METRICS_FILE = REPORTS_DIR / "p8_metrics.json"
REPORT_FILE = REPORTS_DIR / "p8_report.md"


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
    lineages = load_json(PROCESSED_DIR / "lineage_graph.json")
    rollbacks = load_jsonl(PROCESSED_DIR / "rollback_events.jsonl")
    alerts = load_jsonl(PROCESSED_DIR / "alerts.jsonl")
    audit_log = load_jsonl(PROCESSED_DIR / "audit_log.jsonl")
    incidents = load_jsonl(PROCESSED_DIR / "incident_reviews.jsonl")
    sla_report = load_json(PROCESSED_DIR / "sla_report.json")
    ui_panels = load_json(PROCESSED_DIR.parent / "console" / "ui_panels.json")

    regressed_runs = [item for item in experiments if item["status"] == "regressed"]
    failed_runs = [item for item in experiments if item["status"] == "failed"]
    metrics = {
        "tenant_count": len(scope["tenants"]),
        "project_count": len(scope["projects"]),
        "role_count": len(scope["roles"]),
        "layer_count": len(architecture["layers"]),
        "api_count": len(apis),
        "queue_count": len(queues),
        "ui_panel_count": len(ui_panels),
        "dataset_version_count": len(versions),
        "released_version_count": sum(item["status"] == "released" for item in versions),
        "experiment_count": len(experiments),
        "experiment_status_distribution": dict(Counter(item["status"] for item in experiments)),
        "regression_run_count": len(regressed_runs),
        "failed_run_count": len(failed_runs),
        "rollback_count": len(rollbacks),
        "alert_count": len(alerts),
        "resolved_alert_rate": round(sum(item["status"] == "resolved" for item in alerts) / max(1, len(alerts)), 4),
        "audit_event_count": len(audit_log),
        "incident_count": len(incidents),
        "avg_incident_mttr_minutes": round(sum(item["mttr_minutes"] for item in incidents) / max(1, len(incidents)), 2),
        "sla_met_rate": round(sum(value == "met" for value in sla_report["status"].values()) / max(1, len(sla_report["status"])), 4),
        "lineage_node_count": len(lineages["nodes"]),
        "lineage_edge_count": len(lineages["edges"]),
        "governance_workflow_count": len(governance["standard_workflows"]),
        "raci_workstream_count": len(operating_model["raci_matrix"]),
        "oncall_tier_count": len(operating_model["oncall_rotation"]),
        "operating_cadence_count": len(operating_model["operating_cadence"]),
    }
    write_json(metrics, METRICS_FILE)

    report = f"""# P8 Enterprise DataOps Platform Report

## 1. 平台目标与范围

- 租户数：{metrics['tenant_count']}
- 项目数：{metrics['project_count']}
- 角色数：{metrics['role_count']}
- API 数：{metrics['api_count']}

## 2. 核心架构设计

- 核心层数：{metrics['layer_count']}
- 队列数：{metrics['queue_count']}
- UI 面板数：{metrics['ui_panel_count']}
- 血缘节点/边：{metrics['lineage_node_count']}/{metrics['lineage_edge_count']}

## 3. 版本与实验链路落地

- 数据版本数：{metrics['dataset_version_count']}
- 已发布版本数：{metrics['released_version_count']}
- 实验数：{metrics['experiment_count']}
- 实验状态分布：{metrics['experiment_status_distribution']}
- 回滚事件数：{metrics['rollback_count']}

## 4. 可观测性与运营体系

- 告警数：{metrics['alert_count']}
- 已解决告警比例：{metrics['resolved_alert_rate']}
- SLA 达标率：{metrics['sla_met_rate']}
- 平均事故恢复时长：{metrics['avg_incident_mttr_minutes']} 分钟

## 5. 组织协同与治理

- 审计事件数：{metrics['audit_event_count']}
- 事故复盘数：{metrics['incident_count']}
- 标准流程数：{metrics['governance_workflow_count']}
- RACI 工作流数：{metrics['raci_workstream_count']}
- 值班层级数：{metrics['oncall_tier_count']}
- 运营节奏数：{metrics['operating_cadence_count']}

## 6. 扩展方向

- 从单团队平台扩展到跨 BU 共享平台和统一资产目录。
- 引入实时特征、成本预算守卫和跨区域容灾切换。
- 将审批、例外申请和策略引擎进一步自动化。
"""
    REPORT_FILE.write_text(report, encoding="utf-8")
    print("✅ P8 报告生成完成。")
    print(report)


if __name__ == "__main__":
    main()
