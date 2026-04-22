from __future__ import annotations

from collections import Counter

from pipeline_utils import PROCESSED_DIR, QA_VIZ_DIR, REPORTS_DIR, TRAINING_DIR, ensure_dir, load_json, load_jsonl, write_json

METRICS_FILE = REPORTS_DIR / "p3_metrics.json"
REPORT_FILE = REPORTS_DIR / "p3_report.md"


def main() -> None:
    ensure_dir(REPORTS_DIR)
    assets = load_jsonl(PROCESSED_DIR / "asset_manifest.jsonl")
    llava_instruct = load_jsonl(PROCESSED_DIR / "llava_instruct.jsonl")
    alignment = load_jsonl(PROCESSED_DIR / "llava_alignment.jsonl")
    interleaved = load_jsonl(PROCESSED_DIR / "llava_interleaved.jsonl")
    quality = load_json(PROCESSED_DIR / "quality_summary.json")
    manifest = load_json(TRAINING_DIR / "training_manifest.json")

    metrics = {
        "num_assets_total": len(assets),
        "asset_type_distribution": dict(Counter(asset["asset_type"] for asset in assets)),
        "num_base_records": len(llava_instruct),
        "num_alignment_records": len(alignment),
        "num_interleaved_records": len(interleaved),
        "quality_pass_rate": round(quality["passed_records"] / max(1, quality["total_records"]), 4),
        "qa_visualizations": len(list(QA_VIZ_DIR.glob("viz_*"))),
        "training_manifest": manifest,
        "estimated_manual_review_hours": round(quality["total_records"] * 0.5 / 60, 2),
        "estimated_manual_review_cost_rmb": round(quality["total_records"] * 0.5 * 120 / 60, 2),
        "estimated_external_caption_cost_usd": round(len(assets) * 0.015, 2),
    }
    write_json(metrics, METRICS_FILE)

    report = f"""# P3 LLaVA Multimodal Instruction Factory Report

## 1. 项目背景与目标

- 构建一个小型、可复现的多模态指令数据工厂，覆盖图像描述、区域定位、文档阅读和多图比较。
- 当前样本边界：基于 29 张本地图像及其 COCO 标注，扩展出通用图像、文档风格图像和图表风格图像三类资产。

## 2. 图像采集与重描述

- 原始通用图像：{metrics["asset_type_distribution"].get("general_image", 0)}
- 派生文档图像：{metrics["asset_type_distribution"].get("document_image", 0)}
- 派生图表图像：{metrics["asset_type_distribution"].get("chart_image", 0)}
- 总资产数：{metrics["num_assets_total"]}

## 3. 指令构造与对齐

- 基础指令样本：{metrics["num_base_records"]}
- 区域对齐样本：{metrics["num_alignment_records"]}
- 多图交错样本：{metrics["num_interleaved_records"]}
- 最终训练样本：{manifest["num_records"]}
- 任务分布：{manifest["task_distribution"]}

## 4. 质量控制与抽检

- 规则校验通过率：{metrics["quality_pass_rate"]}
- 质量审计样本数：{quality["total_records"]}
- 低质量样本数：{quality["failed_records"]}
- BBox 可视化抽检图：{metrics["qa_visualizations"]}

## 5. 结果展示与工程经验

- 训练集划分：train={manifest["num_train_records"]} val={manifest["num_val_records"]} smoke={manifest["num_smoke_records"]}
- 估算总 token 数：{manifest["estimated_tokens_total"]}
- 参考外部 caption 重写成本估算：{metrics["estimated_external_caption_cost_usd"]} USD
- 人工抽检工时估算：{metrics["estimated_manual_review_hours"]} 小时
- 人工抽检成本估算：{metrics["estimated_manual_review_cost_rmb"]} 元

## 6. 扩展方向

- 接入真实 OCR、表格和图表数据源，替换当前的派生文档/图表样本。
- 增加更长上下文的图文交错样本，扩展到视频帧和文档页序列。
- 引入教师模型重写与人工复核闭环，升级为更高质量的多模态 SFT 工厂。
"""

    REPORT_FILE.write_text(report, encoding="utf-8")
    print("✅ P3 报告生成完成。")
    print(report)


if __name__ == "__main__":
    main()
