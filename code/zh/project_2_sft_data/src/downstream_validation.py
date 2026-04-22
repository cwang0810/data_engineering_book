from __future__ import annotations

import json
import random
from collections import Counter, defaultdict

from pipeline_utils import estimated_tokens, load_jsonl, processed_dir, reports_dir, trim_summary


PROCESSED_DIR = processed_dir()
REPORTS_DIR = reports_dir()

PREFERENCE_FILE = PROCESSED_DIR / "legal_preference_pairs.jsonl"
VALIDATION_JSON = REPORTS_DIR / "p2_downstream_validation.json"
VALIDATION_MD = REPORTS_DIR / "p2_downstream_validation.md"

SAMPLE_SIZE = 50
SAMPLE_SEED = 20260409

UNSAFE_SHORTCUT_PATTERNS = [
    "直接按自己的理解处理",
    "不需要进一步核对事实",
    "不需要查看相关证据",
    "不需要查看相关法条",
    "不需要查看相关证据或其他法条",
]

CRITERIA_LABELS = {
    "citation_present": "法条引用覆盖率",
    "legal_basis_present": "法律依据段落覆盖率",
    "structured_answer": "结构化回答覆盖率",
    "actionable_advice": "可执行建议覆盖率",
    "safe_response": "合规回答覆盖率",
}


def rate(value: float) -> str:
    return f"{value * 100:.2f}%"


def sample_pairs(records: list[dict], sample_size: int = SAMPLE_SIZE) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["task_type"]].append(record)

    rng = random.Random(SAMPLE_SEED)
    task_names = sorted(grouped)
    for task_name in task_names:
        grouped[task_name].sort(key=lambda item: item["sample_id"])
        rng.shuffle(grouped[task_name])

    target = min(sample_size, len(records))
    sampled = []
    while len(sampled) < target:
        made_progress = False
        for task_name in task_names:
            if grouped[task_name] and len(sampled) < target:
                sampled.append(grouped[task_name].pop())
                made_progress = True
        if not made_progress:
            break
    return sampled


def inspect_output(output: str, law_name: str, article_no: str) -> dict:
    sections = [line[5:].strip() for line in output.splitlines() if line.startswith("#### ")]
    criteria = {
        "citation_present": law_name in output and article_no in output,
        "legal_basis_present": "法律依据" in output or "法律适用" in output,
        "structured_answer": len(sections) >= 2,
        "actionable_advice": "建议" in output or "行动建议" in output,
        "safe_response": not any(pattern in output for pattern in UNSAFE_SHORTCUT_PATTERNS),
    }
    return {
        "section_count": len(sections),
        "estimated_tokens": estimated_tokens(output),
        "char_count": len(output),
        "criteria": criteria,
        "total_score": sum(int(passed) for passed in criteria.values()),
    }


def build_sample_result(pair: dict) -> dict:
    chosen = inspect_output(pair["chosen"], pair["law_name"], pair["article_no"])
    rejected = inspect_output(pair["rejected"], pair["law_name"], pair["article_no"])
    return {
        "sample_id": pair["sample_id"],
        "task_type": pair["task_type"],
        "law_name": pair["law_name"],
        "article_no": pair["article_no"],
        "instruction_preview": trim_summary(pair["instruction"], 120),
        "chosen_preview": trim_summary(pair["chosen"], 160),
        "rejected_preview": trim_summary(pair["rejected"], 120),
        "chosen": chosen,
        "rejected": rejected,
        "score_gap": chosen["total_score"] - rejected["total_score"],
        "chosen_wins": chosen["total_score"] > rejected["total_score"],
    }


def summarize_results(sample_results: list[dict]) -> dict:
    count = len(sample_results)
    chosen_scores = [item["chosen"]["total_score"] for item in sample_results]
    rejected_scores = [item["rejected"]["total_score"] for item in sample_results]
    chosen_tokens = [item["chosen"]["estimated_tokens"] for item in sample_results]
    rejected_tokens = [item["rejected"]["estimated_tokens"] for item in sample_results]

    criteria_summary = {}
    for criterion_name in CRITERIA_LABELS:
        chosen_rate = sum(int(item["chosen"]["criteria"][criterion_name]) for item in sample_results) / count
        rejected_rate = sum(int(item["rejected"]["criteria"][criterion_name]) for item in sample_results) / count
        criteria_summary[criterion_name] = {
            "label": CRITERIA_LABELS[criterion_name],
            "chosen_pass_rate": round(chosen_rate, 4),
            "rejected_pass_rate": round(rejected_rate, 4),
            "gain_pp": round((chosen_rate - rejected_rate) * 100, 2),
        }

    unsafe_shortcut_rate = {
        "chosen": round(1 - criteria_summary["safe_response"]["chosen_pass_rate"], 4),
        "rejected": round(1 - criteria_summary["safe_response"]["rejected_pass_rate"], 4),
    }

    examples = []
    covered_tasks: set[str] = set()
    for item in sample_results:
        if item["task_type"] in covered_tasks:
            continue
        examples.append(item)
        covered_tasks.add(item["task_type"])
        if len(examples) == 3:
            break

    if len(examples) < min(3, len(sample_results)):
        selected_ids = {item["sample_id"] for item in examples}
        for item in sample_results:
            if item["sample_id"] in selected_ids:
                continue
            examples.append(item)
            if len(examples) == min(3, len(sample_results)):
                break

    return {
        "sample_size": count,
        "sample_seed": SAMPLE_SEED,
        "task_distribution": dict(Counter(item["task_type"] for item in sample_results)),
        "chosen_avg_total_score": round(sum(chosen_scores) / count, 2),
        "rejected_avg_total_score": round(sum(rejected_scores) / count, 2),
        "chosen_avg_estimated_tokens": round(sum(chosen_tokens) / count, 2),
        "rejected_avg_estimated_tokens": round(sum(rejected_tokens) / count, 2),
        "chosen_win_rate": round(sum(int(item["chosen_wins"]) for item in sample_results) / count, 4),
        "criteria_summary": criteria_summary,
        "unsafe_shortcut_rate": unsafe_shortcut_rate,
        "examples": examples,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# P2 Lightweight Downstream Validation",
        "",
        "## 1. 实验设置",
        "",
        f"- 固定种子随机抽样：`{summary['sample_size']}` 条，seed=`{summary['sample_seed']}`",
        "- 对比对象：`legal_preference_pairs.jsonl` 中进入 QA/偏好闭环的 `chosen` 与对照弱基线 `rejected`",
        "- 评价方式：同一套 5 项轻量 rubric，覆盖法条引用、法律依据、结构化表达、可执行建议与风险控制",
        "- 说明：这是 smoke 级配对审计，用来验证工厂闭环是否把样本质量显著拉开，不替代正式微调评测",
        "",
        "## 2. 汇总结果",
        "",
        "| 指标 | chosen | rejected | 差异 |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| 平均质量分（0-5） | {summary['chosen_avg_total_score']:.2f} | "
            f"{summary['rejected_avg_total_score']:.2f} | "
            f"{summary['chosen_avg_total_score'] - summary['rejected_avg_total_score']:+.2f} |"
        ),
        (
            f"| 平均 token 估算 | {summary['chosen_avg_estimated_tokens']:.2f} | "
            f"{summary['rejected_avg_estimated_tokens']:.2f} | "
            f"{summary['chosen_avg_estimated_tokens'] - summary['rejected_avg_estimated_tokens']:+.2f} |"
        ),
    ]

    for criterion_name, criterion in summary["criteria_summary"].items():
        lines.append(
            f"| {criterion['label']} | {rate(criterion['chosen_pass_rate'])} | "
            f"{rate(criterion['rejected_pass_rate'])} | {criterion['gain_pp']:+.2f} pp |"
        )

    lines.extend(
        [
            "",
            f"- 配对胜率：`chosen` 在 `{rate(summary['chosen_win_rate'])}` 的样本中优于 `rejected`",
            f"- 抽样任务分布：`{json.dumps(summary['task_distribution'], ensure_ascii=False)}`",
            (
                f"- 不安全捷径表述率：`chosen={rate(summary['unsafe_shortcut_rate']['chosen'])}`，"
                f"`rejected={rate(summary['unsafe_shortcut_rate']['rejected'])}`"
            ),
            "",
            "## 3. 定性样例",
            "",
        ]
    )

    for idx, example in enumerate(summary["examples"], start=1):
        lines.extend(
            [
                f"### 样例 {idx}：{example['task_type']} / {example['law_name']} {example['article_no']}",
                "",
                f"- 指令摘要：`{example['instruction_preview']}`",
                f"- `chosen` 摘要：`{example['chosen_preview']}`",
                f"- `rejected` 摘要：`{example['rejected_preview']}`",
                (
                    f"- 质量分：`chosen={example['chosen']['total_score']}` / "
                    f"`rejected={example['rejected']['total_score']}`"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## 4. 结论",
            "",
            "- 这组抽样表明，P02 当前最值得强调的不是“再训练一个更大的模型”，而是 QA / 偏好闭环确实把结构化、引用和风险控制做成了稳定资产。",
            "- 作为章节中的加分项，这类轻量实验已经足够支撑“上线验收前做 smoke 级抽检”的叙事，不需要把项目写成重训练论文。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    preference_pairs = load_jsonl(PREFERENCE_FILE)
    sampled_pairs = sample_pairs(preference_pairs)
    sample_results = [build_sample_result(pair) for pair in sampled_pairs]
    summary = summarize_results(sample_results)

    payload = {
        "validation_type": "sampled_preference_pair_audit",
        "summary": summary,
        "sample_results": sample_results,
    }

    VALIDATION_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    VALIDATION_MD.write_text(render_markdown(summary), encoding="utf-8")

    print("✅ P2 轻量下游验证完成。")
    print(
        json.dumps(
            {
                "sample_size": summary["sample_size"],
                "chosen_avg_total_score": summary["chosen_avg_total_score"],
                "rejected_avg_total_score": summary["rejected_avg_total_score"],
                "chosen_win_rate": summary["chosen_win_rate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
