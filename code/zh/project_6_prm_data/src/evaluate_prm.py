from __future__ import annotations

from collections import Counter

from pipeline_utils import PROCESSED_DIR, REPORTS_DIR, TRAINING_DIR, ensure_standard_dirs, load_json, load_jsonl, write_json

METRICS_FILE = REPORTS_DIR / "p6_metrics.json"
REPORT_FILE = REPORTS_DIR / "p6_report.md"


def main() -> None:
    ensure_standard_dirs()
    seeds = load_jsonl(PROCESSED_DIR / "seed_pool.jsonl")
    traces = load_jsonl(PROCESSED_DIR / "validated_traces.jsonl")
    step_rewards = load_jsonl(PROCESSED_DIR / "step_rewards.jsonl")
    validation_summary = load_json(PROCESSED_DIR / "validation_summary.json")
    manifest = load_json(TRAINING_DIR / "training_manifest.json")

    positive_traces = [trace for trace in traces if trace["trace_type"] == "positive"]
    outcome_only_count = sum(trace["validation_passed"] for trace in traces)
    process_supervision_only_signal = sum(1 for step in step_rewards if step["label"] == 0)

    metrics = {
        "seed_count": len(seeds),
        "trace_count": len(traces),
        "step_count": len(step_rewards),
        "domain_distribution": dict(Counter(trace["domain"] for trace in traces)),
        "trace_type_distribution": dict(Counter(trace["trace_type"] for trace in traces)),
        "reward_bucket_distribution": dict(Counter(trace["reward_bucket"] for trace in traces)),
        "validation_pass_rate": validation_summary["validation_pass_rate"],
        "positive_trace_pass_rate": round(sum(trace["validation_passed"] for trace in positive_traces) / max(1, len(positive_traces)), 4),
        "outcome_only_supervision_count": outcome_only_count,
        "process_supervision_only_signal_steps": process_supervision_only_signal,
        "training_manifest": manifest,
        "estimated_manual_review_hours": round(len(traces) * 1.5 / 60, 2),
        "estimated_manual_review_cost_rmb": round(len(traces) * 1.5 * 100 / 60, 2),
        "estimated_external_generation_cost_usd": 0.0,
    }
    write_json(metrics, METRICS_FILE)

    report = f"""# P6 CoT and PRM Data Factory Report

## 1. 项目目标与任务设定

- 构建一个小型、可复现的 CoT 推理数据与 PRM 训练数据工厂。
- 任务覆盖：数学推理与代码推理两类任务。
- 监督目标：同时关注最终结果正确性与 step-level 过程质量。

## 2. 推理轨迹生成

- 种子数：{metrics['seed_count']}
- 轨迹总数：{metrics['trace_count']}
- 轨迹类型分布：{metrics['trace_type_distribution']}
- 领域分布：{metrics['domain_distribution']}

## 3. 自动验证与打分

- 验证通过率：{metrics['validation_pass_rate']}
- 正例轨迹通过率：{metrics['positive_trace_pass_rate']}
- 奖励桶分布：{metrics['reward_bucket_distribution']}
- 过程监督专有信号步数：{metrics['process_supervision_only_signal_steps']}

## 4. PRM 训练数据组织

- Step-level 样本数：{metrics['step_count']}
- 训练切分：train={manifest['num_train_records']} val={manifest['num_val_records']} smoke={manifest['num_smoke_records']}
- 标签分布：{manifest['label_distribution']}
- 奖励桶分布：{manifest['reward_bucket_distribution']}

## 5. 实验结果与复盘

- 仅结果监督可直接利用的轨迹数：{metrics['outcome_only_supervision_count']}
- 过程监督额外保留的错误步骤监督信号：{metrics['process_supervision_only_signal_steps']}
- 估算总 token 数：{manifest['estimated_tokens_total']}
- 人工抽检工时估算：{metrics['estimated_manual_review_hours']} 小时
- 人工抽检成本估算：{metrics['estimated_manual_review_cost_rmb']} 元

## 6. 扩展方向

- 扩展到科学推理、表格推理与代理规划任务。
- 引入更强的规则和执行器，做 finer-grained step reward 标注。
- 将 PRM 监督与在线采样闭环结合，支持更长轨迹的主动纠错。
"""

    REPORT_FILE.write_text(report, encoding="utf-8")
    print("✅ P6 报告生成完成。")
    print(report)


if __name__ == "__main__":
    main()
