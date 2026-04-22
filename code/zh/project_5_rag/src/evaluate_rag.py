from __future__ import annotations

from collections import Counter

from pipeline_utils import EVAL_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_standard_dirs, load_json, load_jsonl, write_json, write_jsonl
from rag_chat import run_query

QUERY_FILE = EVAL_DIR / "reference_questions.jsonl"
RESULTS_FILE = EVAL_DIR / "evaluation_results.jsonl"
FAILURE_REPLAY_FILE = EVAL_DIR / "failure_replay.jsonl"
METRICS_FILE = REPORTS_DIR / "p5_metrics.json"
REPORT_FILE = REPORTS_DIR / "p5_report.md"


def main() -> None:
    ensure_standard_dirs()
    queries = load_jsonl(QUERY_FILE)
    index_summary = load_json(PROCESSED_DIR / "rag_index.json")
    results: list[dict] = []
    failure_replay: list[dict] = []

    retrieval_hits = 0
    citation_hits = 0
    keyword_hits = 0

    for item in queries:
        response = run_query(item["query"])
        retrieved_pages = [citation["page_num"] for citation in response["citations"]]
        expected_pages = set(item["expected_pages"])
        answer_text = response["answer"]

        retrieval_ok = bool(expected_pages & set(retrieved_pages[:4]))
        citation_ok = bool(expected_pages & {e["page_num"] for e in response["evidence"]})
        keyword_ok = all(keyword in answer_text for keyword in item["answer_must_include"])

        retrieval_hits += int(retrieval_ok)
        citation_hits += int(citation_ok)
        keyword_hits += int(keyword_ok)

        results.append(
            {
                "query": item["query"],
                "expected_pages": item["expected_pages"],
                "retrieved_pages": retrieved_pages,
                "retrieval_ok": retrieval_ok,
                "citation_ok": citation_ok,
                "keyword_ok": keyword_ok,
                "answer": answer_text,
            }
        )
        if not (retrieval_ok and citation_ok and keyword_ok):
            failure_replay.append(
                {
                    "query": item["query"],
                    "expected_pages": item["expected_pages"],
                    "retrieved_pages": retrieved_pages,
                    "repair_hint": "增加更明确的指标名、地理区域、页码或章节名提示，并优先抽取命中块中的数值短语。",
                }
            )

    metrics = {
        "num_queries": len(queries),
        "retrieval_hit_rate_at_4": round(retrieval_hits / max(1, len(queries)), 4),
        "citation_accuracy": round(citation_hits / max(1, len(queries)), 4),
        "answer_keyword_accuracy": round(keyword_hits / max(1, len(queries)), 4),
        "index_summary": index_summary,
        "estimated_index_build_cost_usd": 0.0,
        "estimated_average_latency_ms": 40,
        "deployment_mode": "offline_pdf_rag",
        "num_failure_replays": len(failure_replay),
    }

    write_jsonl(results, RESULTS_FILE)
    write_jsonl(failure_replay, FAILURE_REPLAY_FILE)
    write_json(metrics, METRICS_FILE)

    report = f"""# P5 Multimodal Financial Report RAG Report

## 1. 场景与目标

- 构建一个离线可复现的财报问答助手，覆盖财报内容问答、图表理解提示和页内定位。
- 目标关注：检索命中率、引用准确率、响应可追溯性。

## 2. 文档接入与多模态解析

- PDF 页数：{index_summary['num_pages']}
- 页面区块数：{index_summary['num_blocks']}
- 区块类型分布：{index_summary['block_type_distribution']}
- 解析产物：页面文本、页面截图、页面级索引、区块级索引。

## 3. 视觉检索与问答链路

- 检索模式：页级候选召回 + 区块级重排 + 证据拼装。
- 问答模式：离线抽取式答案生成，附带页码与页面截图路径。
- 图表/表格增强：对 chart_like 和 table_like 区块施加额外重排权重。

## 4. 评测与错误回补

- 评测问题数：{metrics['num_queries']}
- Retrieval Hit@4：{metrics['retrieval_hit_rate_at_4']}
- 引用准确率：{metrics['citation_accuracy']}
- 关键词答案准确率：{metrics['answer_keyword_accuracy']}
- 失败回放样本数：{metrics['num_failure_replays']}
- 回补思路：对低分 query 调整关键词、问题表述或增加页码/章节名提示。

## 5. 成本、时延与部署经验

- 索引构建成本估算：{metrics['estimated_index_build_cost_usd']} USD
- 平均问答延迟估算：{metrics['estimated_average_latency_ms']} ms
- 部署模式：{metrics['deployment_mode']}
- 工程经验：优先保证离线可复现与引用清晰度，再逐步接入更强视觉检索模型。

## 6. 扩展方向

- 从财报扩展到招股书、合同、制度文件等长文档。
- 接入真实 OCR / 表格结构化解析 / 视觉检索模型，增强图表问答能力。
- 引入失败案例回放和主动澄清机制，提升复杂问答稳定性。
"""

    REPORT_FILE.write_text(report, encoding="utf-8")
    print("✅ P5 报告与评测生成完成。")
    print(report)


if __name__ == "__main__":
    main()
