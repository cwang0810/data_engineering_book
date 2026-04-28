from __future__ import annotations

import subprocess
from collections import Counter

from pipeline_utils import (
    EVAL_DIR,
    PAGE_IMAGE_DIR,
    PDF_PATH,
    PROCESSED_DIR,
    chunk_page_blocks,
    classify_block,
    ensure_standard_dirs,
    first_sentences,
    normalize_text,
    split_pages,
    tokenize,
    write_json,
    write_jsonl,
)

TEXT_FALLBACK_PATH = PDF_PATH.with_suffix(".txt")
PAGE_FILE = PROCESSED_DIR / "page_units.jsonl"
BLOCK_FILE = PROCESSED_DIR / "block_units.jsonl"
INDEX_FILE = PROCESSED_DIR / "rag_index.json"
QUERY_FILE = EVAL_DIR / "reference_questions.jsonl"


def run_command(command: list[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    return completed.stdout


def extract_pdf_text() -> list[str]:
    if TEXT_FALLBACK_PATH.exists():
        return split_pages(TEXT_FALLBACK_PATH.read_text(encoding="utf-8"))
    output = run_command(["pdftotext", "-layout", str(PDF_PATH), "-"])
    return split_pages(output)


def render_page_images(num_pages: int) -> None:
    if TEXT_FALLBACK_PATH.exists() and not PDF_PATH.exists():
        from PIL import Image, ImageDraw

        PAGE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        for page_num in range(1, num_pages + 1):
            image = Image.new("RGB", (900, 1200), color=(250, 250, 248))
            draw = ImageDraw.Draw(image)
            draw.text((80, 100), f"Smoke financial report page {page_num}", fill=(30, 41, 59))
            draw.text((80, 160), "Offline text fixture rendered as a placeholder page image.", fill=(75, 85, 99))
            image.save(PAGE_IMAGE_DIR / f"page-{page_num}.png")
        return

    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-f",
            "1",
            "-l",
            str(num_pages),
            str(PDF_PATH),
            str(PAGE_IMAGE_DIR / "page"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def build_reference_questions(pages: list[dict]) -> list[dict]:
    samples = [
        {
            "query": "2024年华为全年实现多少销售收入？",
            "expected_pages": [1, 3, 16, 61],
            "answer_must_include": ["8,621亿", "销售收入"],
        },
        {
            "query": "过去三年华为每年将销售收入的多少投入研究与开发？",
            "expected_pages": [1, 48, 50],
            "answer_must_include": ["20%以上", "研究与开发"],
        },
        {
            "query": "目录里“管理层讨论与分析”在哪一页开始？",
            "expected_pages": [2],
            "answer_must_include": ["12"],
        },
        {
            "query": "在中国，分布式训练性能达到集中式训练性能的多少？",
            "expected_pages": [19],
            "answer_must_include": ["95%以上", "分布式训练"],
        },
        {
            "query": "在中东，AI辅助分析帮助运营商营销转化率提升超过多少？",
            "expected_pages": [20],
            "answer_must_include": ["超过10%", "营销转化率"],
        },
        {
            "query": "在中国，机房改造后PUE从多少优化到多少？",
            "expected_pages": [20],
            "answer_must_include": ["2.0", "1.3"],
        },
        {
            "query": "在中国，智算数据中心总能耗降低约多少？",
            "expected_pages": [20],
            "answer_must_include": ["约10%", "总能耗"],
        },
        {
            "query": "华为在撒哈拉沙漠边缘帮助建设了连续覆盖多少公里的农网站点？",
            "expected_pages": [21],
            "answer_must_include": ["700公里", "农网站点"],
        },
    ]
    return samples


def main() -> None:
    ensure_standard_dirs()
    pages_text = extract_pdf_text()
    render_page_images(len(pages_text))

    page_records: list[dict] = []
    block_records: list[dict] = []
    for page_num, page_text in enumerate(pages_text, start=1):
        image_path = PAGE_IMAGE_DIR / f"page-{page_num}.png"
        page_type_counts = Counter()
        page_blocks = chunk_page_blocks(page_text)
        for block_index, block_text in enumerate(page_blocks, start=1):
            block_type = classify_block(block_text)
            page_type_counts[block_type] += 1
            block_records.append(
                {
                    "block_id": f"p{page_num}_b{block_index}",
                    "page_num": page_num,
                    "block_index": block_index,
                    "block_type": block_type,
                    "text": block_text,
                    "tokens": tokenize(block_text),
                    "preview": first_sentences(block_text, limit=1),
                }
            )

        page_records.append(
            {
                "page_num": page_num,
                "text": normalize_text(page_text),
                "image_path": image_path.relative_to(PDF_PATH.parent).as_posix(),
                "tokens": tokenize(page_text),
                "block_count": len(page_blocks),
                "block_type_distribution": dict(page_type_counts),
                "preview": first_sentences(page_text, limit=2),
            }
        )

    index_summary = {
        "num_pages": len(page_records),
        "num_blocks": len(block_records),
        "block_type_distribution": dict(Counter(block["block_type"] for block in block_records)),
        "page_image_dir": PAGE_IMAGE_DIR.relative_to(PDF_PATH.parent).as_posix(),
    }

    write_jsonl(page_records, PAGE_FILE)
    write_jsonl(block_records, BLOCK_FILE)
    write_json(index_summary, INDEX_FILE)
    write_jsonl(build_reference_questions(page_records), QUERY_FILE)
    print("✅ 财报 PDF 页面与区块索引构建完成。")
    print(index_summary)


if __name__ == "__main__":
    main()
