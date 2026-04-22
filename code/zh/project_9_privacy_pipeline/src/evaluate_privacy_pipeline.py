from __future__ import annotations

import re
from collections import Counter

from pipeline_utils import PROCESSED_DIR, REPORTS_DIR, ensure_standard_dirs, load_json, load_jsonl, write_json

METRICS_FILE = REPORTS_DIR / "p9_metrics.json"
REPORT_FILE = REPORTS_DIR / "p9_report.md"

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
BANK_RE = re.compile(r"\b\d{10,12}\b")
PATIENT_RE = re.compile(r"\bPT-\d{4,6}\b")


def has_direct_pii(text: str) -> bool:
    return any(regex.search(text) for regex in [EMAIL_RE, PHONE_RE, SSN_RE, BANK_RE, PATIENT_RE])


def main() -> None:
    ensure_standard_dirs()
    scope = load_json(PROCESSED_DIR / "compliance_scope.json")
    classification = load_json(PROCESSED_DIR / "classification_policy.json")
    access = load_json(PROCESSED_DIR / "access_policy.json")
    tech_options = load_json(PROCESSED_DIR / "privacy_tech_options.json")
    raw_records = load_jsonl(PROCESSED_DIR / "raw_sensitive_records.jsonl")
    classified = load_jsonl(PROCESSED_DIR / "classified_records.jsonl")
    redacted = load_jsonl(PROCESSED_DIR / "redacted_records.jsonl")
    quarantined = load_jsonl(PROCESSED_DIR / "quarantine_records.jsonl")
    alerts = load_jsonl(PROCESSED_DIR / "access_alerts.jsonl")
    audit_log = load_jsonl(PROCESSED_DIR / "audit_log.jsonl")
    preflight = load_json(PROCESSED_DIR / "preflight_checklist.json")
    incident = load_json(PROCESSED_DIR / "incident_simulation.json")
    postmortem = load_json(PROCESSED_DIR / "postmortem_report.json")

    direct_pii_removed_rate = round(
        sum(not has_direct_pii(item["payload"]) for item in redacted) / max(1, len(redacted)),
        4,
    )
    metrics = {
        "domain_count": len(scope["example_domains"]),
        "compliance_target_count": len(scope["compliance_targets"]),
        "source_type_count": len(classification["source_types"]),
        "role_count": len(access["roles"]),
        "privacy_tech_count": len(tech_options),
        "raw_record_count": len(raw_records),
        "restricted_record_count": sum(item["sensitivity_level"] == "restricted" for item in classified),
        "quarantine_count": len(quarantined),
        "pii_detection_distribution": dict(Counter(detection["pattern_name"] for item in classified for detection in item["pii_detections"])),
        "direct_pii_removed_rate": direct_pii_removed_rate,
        "alert_count": len(alerts),
        "resolved_alert_rate": round(sum(item["status"] == "resolved" for item in alerts) / max(1, len(alerts)), 4),
        "audit_event_count": len(audit_log),
        "preflight_pass_rate": round(preflight["passed_checks"] / max(1, preflight["total_checks"]), 4),
        "incident_response_minutes": incident["response_minutes"],
        "postmortem_follow_up_count": len(postmortem["follow_ups"]),
    }
    write_json(metrics, METRICS_FILE)

    report = f"""# P9 Privacy-Preserving Data Pipeline Report

## 1. 场景与合规目标

- 场景域数：{metrics['domain_count']}
- 合规目标数：{metrics['compliance_target_count']}
- 数据源类型数：{metrics['source_type_count']}
- 角色数：{metrics['role_count']}

## 2. 数据分类与权限设计

- 原始记录数：{metrics['raw_record_count']}
- Restricted 记录数：{metrics['restricted_record_count']}
- 隔离/隔离待审记录数：{metrics['quarantine_count']}

## 3. 脱敏、隔离与安全处理

- PII 检测分布：{metrics['pii_detection_distribution']}
- 直接 PII 去除率：{metrics['direct_pii_removed_rate']}
- 告警数：{metrics['alert_count']}
- 审计事件数：{metrics['audit_event_count']}

## 4. 隐私保护技术接入

- 隐私技术选项数：{metrics['privacy_tech_count']}
- 当前演示覆盖差分隐私、TEE、FHE 与 tokenization 接入说明。

## 5. 上线检查与复盘

- 预上线检查通过率：{metrics['preflight_pass_rate']}
- 事故响应时长：{metrics['incident_response_minutes']} 分钟
- 复盘后续动作数：{metrics['postmortem_follow_up_count']}

## 6. 扩展方向

- 从单条流水线扩展到跨系统隐私编排与跨组织协作。
- 引入更细粒度 purpose-based access control 和数据保留策略。
- 将隐私预算与导出审批打通到更完整的数据平台控制面。
"""
    REPORT_FILE.write_text(report, encoding="utf-8")
    print("✅ P9 报告生成完成。")
    print(report)


if __name__ == "__main__":
    main()
