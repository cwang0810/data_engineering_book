from __future__ import annotations

import argparse
import json
from collections import defaultdict

from pipeline_utils import (
    PAGE_IMAGE_DIR,
    PROCESSED_DIR,
    detect_query_type,
    extract_highlight,
    extract_query_terms,
    first_sentences,
    keyword_score,
    load_jsonl,
    normalize_text,
    tokenize,
)

PAGE_FILE = PROCESSED_DIR / "page_units.jsonl"
BLOCK_FILE = PROCESSED_DIR / "block_units.jsonl"


def load_index() -> tuple[list[dict], list[dict]]:
    return load_jsonl(PAGE_FILE), load_jsonl(BLOCK_FILE)


def rank_pages(query: str, pages: list[dict], blocks: list[dict], top_k: int = 4) -> tuple[list[dict], list[dict]]:
    query_tokens = tokenize(query)
    query_terms = extract_query_terms(query)
    query_type = detect_query_type(query)

    page_scores: dict[int, int] = defaultdict(int)
    for page in pages:
        base = keyword_score(query_tokens, page["text"])
        for term in query_terms:
            if term in page["text"]:
                base += len(term) * 5
        if query_type == "chart":
            base += page["block_type_distribution"].get("chart_like", 0) * 3
        elif query_type == "table":
            base += page["block_type_distribution"].get("table_like", 0) * 3
        page_scores[page["page_num"]] += base

    ranked_blocks = []
    for block in blocks:
        score = keyword_score(query_tokens, block["text"])
        for term in query_terms:
            if term in block["text"]:
                score += len(term) * 6
        if query_type == "chart" and block["block_type"] == "chart_like":
            score += 4
        if query_type == "table" and block["block_type"] == "table_like":
            score += 4
        if score > 0:
            ranked_blocks.append((score, block))
            page_scores[block["page_num"]] += score

    ranked_pages = sorted(
        [page for page in pages if page_scores.get(page["page_num"], 0) > 0],
        key=lambda item: (page_scores[item["page_num"]], -item["page_num"]),
        reverse=True,
    )[:top_k]
    ranked_blocks = [block for _, block in sorted(ranked_blocks, key=lambda item: item[0], reverse=True)[:8]]
    return ranked_pages, ranked_blocks


def synthesize_answer(query: str, ranked_pages: list[dict], ranked_blocks: list[dict]) -> dict:
    if not ranked_pages:
        return {
            "query": query,
            "answer": "未检索到明显相关的页面，请尝试换一种表述，或直接询问指标名、页码、章节名。",
            "citations": [],
            "evidence": [],
        }

    query_terms = extract_query_terms(query)
    evidence = []
    seen_pages: set[int] = set()
    for block in ranked_blocks:
        if block["page_num"] in seen_pages and len(evidence) >= 4:
            continue
        if query_terms and not any(term in block["text"] for term in query_terms):
            continue
        snippet = extract_highlight(block["text"], query_terms)
        evidence.append(
            {
                "page_num": block["page_num"],
                "block_type": block["block_type"],
                "snippet": snippet or "；".join(first_sentences(block["text"], limit=2)),
            }
        )
        seen_pages.add(block["page_num"])
        if len(evidence) >= 4:
            break

    if not evidence:
        for page in ranked_pages[:3]:
            evidence.append(
                {
                    "page_num": page["page_num"],
                    "block_type": "page_preview",
                    "snippet": "；".join(page["preview"]),
                }
            )

    cited_pages = sorted({item["page_num"] for item in evidence} or {page["page_num"] for page in ranked_pages})
    summary_line = ""
    if evidence:
        summary_line = f"核心结论：{evidence[0]['snippet']}"
    bullet_lines = [f"- 第{item['page_num']}页：{item['snippet']}" for item in evidence[:4]]
    answer = (
        "基于检索到的财报页面，整理出的回答如下：\n"
        + (summary_line + "\n" if summary_line else "")
        + "\n".join(bullet_lines)
        + "\n引用页码："
        + "、".join(f"第{page}页" for page in cited_pages)
    )

    citations = [
        {
            "page_num": page["page_num"],
            "image_path": page["image_path"],
            "preview": page["preview"],
        }
        for page in ranked_pages
    ]
    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "evidence": evidence,
    }


def run_query(query: str) -> dict:
    pages, blocks = load_index()
    ranked_pages, ranked_blocks = rank_pages(query, pages, blocks)
    return synthesize_answer(query, ranked_pages, ranked_blocks)


def interactive_chat() -> None:
    print("=" * 50)
    print("多模态财报助手（离线页级引用版）")
    print("输入 exit 退出")
    print("=" * 50)
    while True:
        query = input("\n>>> 请提问: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        result = run_query(query)
        print("\n" + result["answer"])
        print("\n引用详情：")
        for citation in result["citations"]:
            print(f"- 第{citation['page_num']}页 | {citation['image_path']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="")
    args = parser.parse_args()
    if args.query:
        print(json.dumps(run_query(args.query), ensure_ascii=False, indent=2))
    else:
        interactive_chat()


if __name__ == "__main__":
    main()
