from __future__ import annotations

import json
from collections import Counter

from pipeline_utils import load_jsonl, processed_dir, reports_dir, training_dir


PROCESSED_DIR = processed_dir()
TRAINING_DIR = training_dir()
REPORTS_DIR = reports_dir()

METRICS_FILE = REPORTS_DIR / "p2_metrics.json"
REPORT_FILE = REPORTS_DIR / "p2_report.md"
DOWNSTREAM_VALIDATION_FILE = REPORTS_DIR / "p2_downstream_validation.json"


def count_lines(path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_chunks = load_jsonl(PROCESSED_DIR / "raw_chunks.jsonl")
    accepted = load_jsonl(PROCESSED_DIR / "domain_expert_sft.jsonl")
    rejected = load_jsonl(PROCESSED_DIR / "synthetic_candidates_rejected.jsonl")
    preference_pairs = load_jsonl(PROCESSED_DIR / "legal_preference_pairs.jsonl")
    qa_reviews = load_jsonl(PROCESSED_DIR / "legal_qa_review.jsonl")
    risk_refusals = load_jsonl(PROCESSED_DIR / "legal_risk_refusal_sft.jsonl")
    risk_register = load_jsonl(PROCESSED_DIR / "legal_risk_register.jsonl")
    final_dataset = load_jsonl(TRAINING_DIR / "final_sft_dataset.jsonl")
    manifest = json.loads((TRAINING_DIR / "training_manifest.json").read_text(encoding="utf-8"))
    downstream_validation = (
        json.loads(DOWNSTREAM_VALIDATION_FILE.read_text(encoding="utf-8")) if DOWNSTREAM_VALIDATION_FILE.exists() else {}
    )
    downstream_summary = downstream_validation.get("summary", {})

    task_distribution = Counter(item["task_type"] for item in accepted)
    law_distribution = Counter(item["law_name"] for item in accepted)

    overall_review = round(
        sum(item["review_scores"]["overall"] for item in qa_reviews) / len(qa_reviews), 2
    ) if qa_reviews else 0.0

    human_review_hours = round(len(qa_reviews) * 1.5 / 60, 2)
    human_review_cost = round(human_review_hours * 120, 2)
    template_model_cost = 0.0
    reference_teacher_cost = round(manifest["estimated_tokens_total"] / 1_000_000 * 8.0, 2)

    metrics = {
        "seed_count": len(raw_chunks),
        "accepted_sft_count": len(accepted),
        "rejected_candidate_count": len(rejected),
        "preference_pair_count": len(preference_pairs),
        "qa_review_count": len(qa_reviews),
        "risk_refusal_count": len(risk_refusals),
        "risk_register_count": len(risk_register),
        "final_dataset_count": len(final_dataset),
        "task_distribution": dict(task_distribution),
        "law_distribution": dict(law_distribution),
        "average_review_score": overall_review,
        "downstream_validation": downstream_summary,
        "cost_analysis": {
            "actual_template_model_cost": template_model_cost,
            "reference_external_teacher_cost": reference_teacher_cost,
            "estimated_human_review_hours": human_review_hours,
            "estimated_human_review_cost_rmb": human_review_cost,
        },
    }

    if downstream_summary:
        downstream_section = f"""## 5. 轻量下游验证（50 条抽样）

- 固定种子随机抽样：{downstream_summary['sample_size']} 条，seed={downstream_summary['sample_seed']}
- `chosen` 平均质量分：{downstream_summary['chosen_avg_total_score']} / 5
- `rejected` 平均质量分：{downstream_summary['rejected_avg_total_score']} / 5
- 配对胜率：{downstream_summary['chosen_win_rate'] * 100:.2f}%
- 法条引用覆盖率：chosen={downstream_summary['criteria_summary']['citation_present']['chosen_pass_rate'] * 100:.2f}% / rejected={downstream_summary['criteria_summary']['citation_present']['rejected_pass_rate'] * 100:.2f}%
- 不安全捷径表述率：chosen={downstream_summary['unsafe_shortcut_rate']['chosen'] * 100:.2f}% / rejected={downstream_summary['unsafe_shortcut_rate']['rejected'] * 100:.2f}%

"""
    else:
        downstream_section = """## 5. 轻量下游验证（50 条抽样）

- 未检测到 `p2_downstream_validation.json`，本次报告未纳入轻量下游验证摘要。

"""

    report = f"""# P2 Legal SFT Factory Report

## 1. 场景与目标

- 任务范围：法律问答、法条解释、案例分析。
- 风格目标：专业、稳定、合规，明确法律依据和行动建议。

## 2. 种子数据与指令体系

- 法条种子数：{len(raw_chunks)}
- 指令体系任务分布：{json.dumps(dict(task_distribution), ensure_ascii=False)}
- 法律来源分布：{json.dumps(dict(law_distribution), ensure_ascii=False)}

## 3. 合成扩张与蒸馏协作

- 通过模板教师生成高价值 SFT：{len(accepted)}
- 通过启发式裁判过滤或构造低质量对照：{len(rejected)}
- 偏好对数量：{len(preference_pairs)}

## 4. QA 与偏好增强

- QA 评审记录：{len(qa_reviews)}
- 高风险拒答样本：{len(risk_refusals)}
- 平均 QA 评分：{overall_review}

{downstream_section}## 6. 效果评测与成本核算

- 最终训练集规模：{len(final_dataset)}
- 训练集拆分：train={manifest['num_train_records']} val={manifest['num_val_records']} smoke={manifest['num_smoke_records']}
- 实际模板模型成本：0
- 参考外部教师模型成本估算：{reference_teacher_cost}
- 人工复核工时估算：{human_review_hours} 小时
- 人工复核成本估算：{human_review_cost} 元

## 7. 扩展方向

- 从法律扩展到金融、医疗、税务等垂直领域。
- 将模板教师替换为真实教师模型，并保留同样的裁判与 QA 结构。
- 引入更强的裁判模型和人工抽检闭环。
"""

    METRICS_FILE.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_FILE.write_text(report, encoding="utf-8")

    print("✅ P2 报告生成完成。")
    print(report)


if __name__ == "__main__":
    main()
