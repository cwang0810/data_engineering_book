from __future__ import annotations

from collections import Counter

from pipeline_utils import BOOKS_DIR, PROCESSED_DIR, REPORTS_DIR, TRAINING_DIR, ensure_standard_dirs, load_json, load_jsonl, write_json

METRICS_FILE = REPORTS_DIR / "p4_metrics.json"
REPORT_FILE = REPORTS_DIR / "p4_report.md"


def main() -> None:
    ensure_standard_dirs()
    seed_pool = load_jsonl(PROCESSED_DIR / "seed_pool.jsonl")
    synthesized = load_jsonl(PROCESSED_DIR / "synthetic_textbook_chapters.jsonl")
    verified = load_jsonl(PROCESSED_DIR / "verified_textbook.jsonl")
    failures = load_jsonl(PROCESSED_DIR / "verification_failures.jsonl")
    quality = load_json(PROCESSED_DIR / "quality_summary.json")
    manifest = load_json(TRAINING_DIR / "training_manifest.json")
    curriculum = load_json(PROCESSED_DIR / "curriculum_map.json")
    catalog = load_json(PROCESSED_DIR / "textbook_catalog.json")

    metrics = {
        "seed_count": len(seed_pool),
        "synthesized_count": len(synthesized),
        "verified_count": len(verified),
        "verification_pass_rate": round(len(verified) / max(1, len(synthesized)), 4),
        "verification_failure_count": len(failures),
        "domain_distribution": dict(Counter(record["domain"] for record in verified)),
        "topic_distribution": dict(Counter(record["topic"] for record in verified)),
        "difficulty_distribution": dict(Counter(record["difficulty"] for record in verified)),
        "training_manifest": manifest,
        "estimated_external_generation_cost_usd": round(len(synthesized) * 0.06, 2),
        "estimated_manual_review_hours": round(len(verified) * 3 / 60, 2),
        "estimated_manual_review_cost_rmb": round(len(verified) * 3 * 100 / 60, 2),
        "quality_summary": quality,
        "num_books": len(catalog["books"]),
        "curriculum_volumes": curriculum["volumes"],
    }
    write_json(metrics, METRICS_FILE)

    report = f"""# P4 Synthetic Math and Code Textbook Report

## 1. 项目目标与范围

- 构建一个可复现的小型合成教材工厂，覆盖数学知识讲解与 Python 编程知识讲解。
- 当前读者边界：面向入门到进阶学习者，输出形式包括概念正文、例题、练习、解答与验证片段。

## 2. 种子池与章节模板

- 种子总数：{metrics["seed_count"]}
- 学科分布：{metrics["domain_distribution"]}
- 难度分布：{metrics["difficulty_distribution"]}
- 主题分布：{metrics["topic_distribution"]}

## 3. 生成、验证与纠错

- 合成章节数：{metrics["synthesized_count"]}
- 通过执行与单元测试验证的章节数：{metrics["verified_count"]}
- 验证通过率：{metrics["verification_pass_rate"]}
- 失败样本数：{metrics["verification_failure_count"]}
- 教材卷册数：{metrics["num_books"]}

## 4. 质量风控

- 质量审计总数：{quality["total_records"]}
- 审计通过数：{quality["passed_records"]}
- 低质量样本数：{quality["failed_records"]}
- 风控覆盖：缺标题、缺正文、缺学习目标、缺章末检查、缺单测、执行失败、过短章节、重复章节

## 5. 结果评测与成本

- 最终训练样本：{manifest["num_records"]}
- 训练集划分：train={manifest["num_train_records"]} val={manifest["num_val_records"]} smoke={manifest["num_smoke_records"]}
- 估算总 token 数：{manifest["estimated_tokens_total"]}
- 参考外部生成成本估算：{metrics["estimated_external_generation_cost_usd"]} USD
- 人工抽检工时估算：{metrics["estimated_manual_review_hours"]} 小时
- 人工抽检成本估算：{metrics["estimated_manual_review_cost_rmb"]} 元
- 成册产物：{", ".join(book["title"] for book in catalog["books"])}

## 6. 扩展方向

- 从小型教材章节扩展到题库库、讲义、课程脚本与助教反馈数据。
- 增加更严格的数学求解器和代码裁判器，构建自动纠错闭环。
- 用真实课程大纲替换当前的规则模板，升级章节顺序与先修关系设计。
"""

    REPORT_FILE.write_text(report, encoding="utf-8")
    print("✅ P4 报告生成完成。")
    print(report)


if __name__ == "__main__":
    main()
