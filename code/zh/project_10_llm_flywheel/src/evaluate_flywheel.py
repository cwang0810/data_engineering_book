from __future__ import annotations

from collections import Counter

from pipeline_utils import CONSOLE_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_standard_dirs, load_json, load_jsonl, write_json

METRICS_FILE = REPORTS_DIR / "p10_metrics.json"
REPORT_FILE = REPORTS_DIR / "p10_report.md"


def main() -> None:
    ensure_standard_dirs()
    registry = load_json(PROCESSED_DIR / "upstream_project_registry.json")
    phase_inventory = load_json(PROCESSED_DIR / "phase_inventory.json")
    architecture = load_json(PROCESSED_DIR / "flywheel_architecture.json")
    boundaries = load_json(PROCESSED_DIR / "system_boundaries.json")
    stage_plan = load_json(PROCESSED_DIR / "stage_plan.json")
    runs = load_jsonl(PROCESSED_DIR / "flywheel_runs.jsonl")
    milestones = load_json(CONSOLE_DIR / "milestone_board.json")

    total_manual_review_hours = round(sum(item["estimated_manual_review_hours"] for item in registry), 2)
    total_manual_review_cost_rmb = round(sum(item["estimated_manual_review_cost_rmb"] for item in registry), 2)
    stage_completion_rate = round(sum(item["status"] == "completed" for item in runs) / max(1, len(runs)), 4)
    avg_stage_score = round(sum(item["score"] for item in runs) / max(1, len(runs)), 4)

    bottlenecks = [
        {"name": "foundation_corpus_scale", "severity": "medium", "reason": "P1 final retention is only 17.37%, limiting base corpus growth."},
        {"name": "prm_validation_gap", "severity": "medium", "reason": "P6 validation pass rate is 0.6759, leaving room for stronger trace verification."},
        {"name": "platform_regression_handling", "severity": "low", "reason": "P8 still observed one regressed run and one failed run, so release gates should stay strict."},
    ]
    cost_model = {
        "estimated_manual_review_hours": total_manual_review_hours,
        "estimated_manual_review_cost_rmb": total_manual_review_cost_rmb,
        "shared_platform_benefit": "P8 reduces duplicated release, rollback, and observability work across all downstream pipelines.",
        "reuse_benefit_examples": [
            "P1 corpus and manifests feed multiple downstream data factories.",
            "P6 reasoning feedback and P7 tool-use templates can be reused across application teams.",
            "P8/P9 centralize governance instead of reimplementing it in each project.",
        ],
    }
    org_model = {
        "teams": [
            {"team": "foundation_data_team", "scope": ["p1", "data intake"]},
            {"team": "alignment_factory_team", "scope": ["p2", "p3", "p4", "p6", "p7"]},
            {"team": "application_team", "scope": ["p5", "user feedback", "failure replay"]},
            {"team": "platform_security_team", "scope": ["p8", "p9", "release and privacy controls"]},
        ],
        "governance_mechanisms": [
            "weekly milestone review",
            "release gate with rollback owner",
            "privacy/export approval checkpoint",
            "cross-team bottleneck retro",
        ],
    }
    executive_dashboard = {
        "project_count": len(registry),
        "all_upstream_projects_passed": phase_inventory["all_projects_passed"],
        "stage_completion_rate": stage_completion_rate,
        "avg_stage_score": avg_stage_score,
        "milestone_done_count": sum(item["status"] == "done" for item in milestones),
        "bottleneck_count": len(bottlenecks),
    }

    metrics = {
        "project_count": len(registry),
        "phase_count": len(phase_inventory["phase_distribution"]),
        "interface_count": phase_inventory["interface_count"],
        "all_upstream_projects_passed": phase_inventory["all_projects_passed"],
        "total_upstream_checks": phase_inventory["total_checks"],
        "total_upstream_passed_checks": phase_inventory["total_passed_checks"],
        "architecture_layer_count": len(architecture["layers"]),
        "control_point_count": len(architecture["control_points"]),
        "boundary_rule_count": len(boundaries["risk_boundaries"]),
        "stage_count": len(stage_plan),
        "stage_completion_rate": stage_completion_rate,
        "avg_stage_score": avg_stage_score,
        "milestone_count": len(milestones),
        "bottleneck_count": len(bottlenecks),
        "estimated_manual_review_hours": total_manual_review_hours,
        "estimated_manual_review_cost_rmb": total_manual_review_cost_rmb,
        "phase_distribution": phase_inventory["phase_distribution"],
        "bottlenecks": bottlenecks,
        "cost_model": cost_model,
        "org_model": org_model,
        "executive_dashboard": executive_dashboard,
    }
    write_json(metrics, METRICS_FILE)
    write_json(bottlenecks, PROCESSED_DIR / "bottleneck_analysis.json")
    write_json(cost_model, PROCESSED_DIR / "cost_model.json")
    write_json(org_model, PROCESSED_DIR / "org_operating_model.json")
    write_json(executive_dashboard, CONSOLE_DIR / "executive_dashboard.json")

    report = f"""# P10 End-to-End LLM Data Flywheel Report

## 1. 项目背景与总目标

- 上游项目数：{metrics['project_count']}
- 阶段数：{metrics['stage_count']}
- 阶段覆盖分布：{metrics['phase_distribution']}
- 所有上游项目测试通过：{metrics['all_upstream_projects_passed']}

## 2. 全链路架构与系统边界

- 架构层数：{metrics['architecture_layer_count']}
- 控制点数：{metrics['control_point_count']}
- 治理/风险边界数：{metrics['boundary_rule_count']}
- 关键接口数：{metrics['interface_count']}

## 3. 分步实现：采集到反馈的全链路集成

- 已完成阶段比例：{metrics['stage_completion_rate']}
- 平均阶段评分：{metrics['avg_stage_score']}
- 里程碑数：{metrics['milestone_count']}
- 总上游检查通过：{metrics['total_upstream_passed_checks']}/{metrics['total_upstream_checks']}

## 4. 结果展示与阶段评估

- 关键瓶颈数：{metrics['bottleneck_count']}
- 核心瓶颈：foundation corpus scale, PRM validation gap, platform regression handling
- 执行总盘：executive dashboard 已生成，可用于阶段汇报。

## 5. 成本优化与组织协同

- 估算人工复核工时：{metrics['estimated_manual_review_hours']} 小时
- 估算人工复核成本：{metrics['estimated_manual_review_cost_rmb']} 元
- 平台复用收益：P8/P9 让 release、rollback、privacy 与 audit 能跨项目复用。

## 6. 扩展方向

- 向企业级长期飞轮和课程总结项目扩展。
- 将在线反馈、A/B 实验、成本预算与多 BU 共享控制面进一步打通。
- 作为全书方法集成的总复盘，继续补更真实的生产链路与监控大盘。
"""
    REPORT_FILE.write_text(report, encoding="utf-8")
    print("✅ P10 总飞轮报告生成完成。")
    print(report)


if __name__ == "__main__":
    main()
