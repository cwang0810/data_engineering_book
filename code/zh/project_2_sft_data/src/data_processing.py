from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pdfplumber
from tqdm import tqdm

from pipeline_utils import (
    ARTICLE_RE,
    build_seed_id,
    extract_article_no,
    law_name_from_source,
    normalize_text,
    processed_dir,
    trim_summary,
)


RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = processed_dir()
RAW_CHUNKS_FILE = PROCESSED_DIR / "raw_chunks.jsonl"
SEED_FILE = PROCESSED_DIR / "legal_seed_dataset.jsonl"
TAXONOMY_FILE = PROCESSED_DIR / "instruction_taxonomy.json"


def clean_pdf_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\[\s*\d+(?:[-–,]\d+)*\s*\]", "", text)
    text = re.sub(r"(?:^|\s)[-—–－]\s*\d+\s*[-—–－](?=\s|$)", " ", text)

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"[-—–－\s\d]+", stripped):
            continue
        stripped = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", stripped)
        stripped = re.sub(r"\s+", " ", stripped).strip()
        if stripped:
            lines.append(stripped)

    return normalize_text("\n".join(lines))


def extract_pdf_text(file_path: Path) -> str:
    full_text = []
    with pdfplumber.open(file_path) as pdf:
        for page in tqdm(pdf.pages, desc=f"读取 {file_path.name}", leave=False):
            width, height = page.width, page.height
            bbox = (0, height * 0.05, width, height * 0.95)
            try:
                page_crop = page.crop(bbox=bbox)
                text = page_crop.extract_text()
                if text:
                    full_text.append(text)
            except Exception:
                continue
    return clean_pdf_text("\n".join(full_text))


def split_legal_articles(full_text: str, law_name: str, source_filename: str) -> list[dict]:
    pattern = r"(第[0-9零一二三四五六七八九十百千]+条[\s\S]*?)(?=第[0-9零一二三四五六七八九十百千]+条|$)"
    matches = re.findall(pattern, full_text)
    seeds = []

    for idx, match in enumerate(matches):
        content = re.sub(r"\s+", " ", match).strip()
        if len(content) < 12:
            continue
        article_no = extract_article_no(content)
        seed_id = build_seed_id(law_name, article_no, content)
        seeds.append(
            {
                "id": seed_id,
                "source": source_filename,
                "law_name": law_name,
                "type": "legal_article",
                "article_no": article_no,
                "content": content,
                "summary": trim_summary(content),
                "char_count": len(content),
                "article_index": idx + 1,
            }
        )
    return seeds


def build_taxonomy(seeds: list[dict]) -> dict:
    law_distribution = Counter(seed["law_name"] for seed in seeds)
    return {
        "scenario": "法律领域 SFT 数据工厂",
        "target_tasks": [
            {
                "task_type": "legal_qa",
                "description": "围绕法条的直接问答",
                "output_format": ["问题重述", "法律依据", "结论与建议"],
            },
            {
                "task_type": "statute_explanation",
                "description": "法条解释与适用范围说明",
                "output_format": ["核心概念", "通俗解释", "适用提醒"],
            },
            {
                "task_type": "case_analysis",
                "description": "围绕具体事实场景的分析",
                "output_format": ["事实识别", "争议焦点", "法律适用", "行动建议"],
            },
        ],
        "style_requirements": [
            "结论明确",
            "引用法条",
            "不代替正式律师代理意见",
            "遇到高风险或违法请求时明确拒绝并引导合法路径",
        ],
        "law_distribution": dict(law_distribution),
        "seed_count": len(seeds),
    }


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(RAW_DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {RAW_DATA_DIR}")

    all_seeds = []
    print(f"🚀 开始构建法律种子数据，共 {len(pdf_files)} 个 PDF。")

    for file_path in pdf_files:
        law_name = law_name_from_source(file_path.name)
        full_text = extract_pdf_text(file_path)
        seeds = split_legal_articles(full_text, law_name, file_path.name)
        all_seeds.extend(seeds)
        print(f"⚖️ {file_path.name}: 提取 {len(seeds)} 条法条种子")

    with RAW_CHUNKS_FILE.open("w", encoding="utf-8") as f_raw, \
         SEED_FILE.open("w", encoding="utf-8") as f_seed:
        for seed in all_seeds:
            line = json.dumps(seed, ensure_ascii=False) + "\n"
            f_raw.write(line)
            f_seed.write(line)

    taxonomy = build_taxonomy(all_seeds)
    with TAXONOMY_FILE.open("w", encoding="utf-8") as f:
        json.dump(taxonomy, f, ensure_ascii=False, indent=2)

    print(f"✅ 种子数据构建完成，共 {len(all_seeds)} 条。")
    print(f"💾 输出: {RAW_CHUNKS_FILE}")
    print(f"💾 输出: {SEED_FILE}")
    print(f"💾 输出: {TAXONOMY_FILE}")


if __name__ == "__main__":
    main()
