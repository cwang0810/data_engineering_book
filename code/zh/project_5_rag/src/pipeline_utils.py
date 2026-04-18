from __future__ import annotations

import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PDF_PATH = DATA_DIR / "annual_report_2024_cn.pdf"
PAGE_IMAGE_DIR = DATA_DIR / "page_images"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"
EVAL_DIR = DATA_DIR / "eval"
QUERY_STOP_PHRASES = [
    "是多少", "多少", "哪一页", "在哪一页", "开始", "达到", "帮助", "实现", "过去三年", "每年将",
    "在中国", "在中东", "华为", "全年", "连续覆盖", "约多少", "多少？", "请问",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_standard_dirs() -> None:
    for path in [PAGE_IMAGE_DIR, PROCESSED_DIR, REPORTS_DIR, EVAL_DIR]:
        ensure_dir(path)


def write_json(data, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(records: list[dict], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_pages(text: str) -> list[str]:
    return [page.strip() for page in text.split("\f") if page.strip()]


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text).lower()
    chinese_terms: list[str] = []
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
        chinese_terms.append(sequence)
        max_n = min(4, len(sequence))
        for n in range(2, max_n + 1):
            for start in range(0, len(sequence) - n + 1):
                chinese_terms.append(sequence[start:start + n])
    latin_terms = re.findall(r"[a-z0-9][a-z0-9\-_\.%]{1,}", normalized)
    return list(dict.fromkeys(chinese_terms + latin_terms))


def chunk_page_blocks(page_text: str) -> list[str]:
    blocks = [normalize_text(block) for block in re.split(r"\n\s*\n", page_text) if normalize_text(block)]
    if blocks:
        return blocks
    lines = [normalize_text(line) for line in page_text.splitlines() if normalize_text(line)]
    return lines


def classify_block(text: str) -> str:
    if "注：" in text or "单位：" in text or "说明：" in text:
        return "footnote"
    if text.count("%") >= 2 or text.count("亿元") >= 2 or text.count("人民币") >= 2:
        return "table_like"
    if "图" in text and ("增长" in text or "趋势" in text or "占比" in text):
        return "chart_like"
    if text.startswith("■") or text.startswith("-"):
        return "bullet"
    return "paragraph"


def keyword_score(query_tokens: list[str], text: str) -> int:
    haystack = normalize_text(text).lower()
    score = 0
    for token in query_tokens:
        if token in haystack:
            score += max(1, len(token) // 2)
    return score


def detect_query_type(query: str) -> str:
    text = normalize_text(query)
    if any(word in text for word in ["图", "趋势", "变化", "增长", "下降", "占比"]):
        return "chart"
    if any(word in text for word in ["表", "财务概要", "收入", "利润", "现金流", "研发"]):
        return "table"
    if any(word in text for word in ["页", "定位", "哪一页", "出处", "引用"]):
        return "locator"
    return "general"


def first_sentences(text: str, limit: int = 2) -> list[str]:
    segments = re.split(r"[。！？;\n]", text)
    cleaned = [normalize_text(segment) for segment in segments if normalize_text(segment)]
    return cleaned[:limit]


def split_clauses(text: str) -> list[str]:
    segments = re.split(r"[。！？；;\n]", text)
    cleaned = [normalize_text(segment) for segment in segments if normalize_text(segment)]
    return cleaned


def contains_numeric_fact(text: str) -> bool:
    return bool(re.search(r"\d[\d,\.]*(?:%|亿|万|人民币|百万元|公里|页|小时|分钟)?", text))


def extract_highlight(text: str, terms: list[str], window: int = 90) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""

    best_snippet = ""
    best_score = -1
    for term in terms:
        for match in re.finditer(re.escape(term), normalized):
            start = max(0, match.start() - window)
            end = min(len(normalized), match.end() + window)
            snippet = normalized[start:end]
            score = sum(len(item) * 3 for item in terms if item in snippet)
            if contains_numeric_fact(snippet):
                score += 10
            if score > best_score:
                best_score = score
                best_snippet = snippet

    if best_snippet:
        return best_snippet

    clauses = split_clauses(normalized)
    if clauses:
        ranked = sorted(
            clauses,
            key=lambda clause: (
                sum(len(term) * 3 for term in terms if term in clause) + (10 if contains_numeric_fact(clause) else 0),
                len(clause),
            ),
            reverse=True,
        )
        return ranked[0]
    return normalized[:window * 2]


def extract_query_terms(query: str) -> list[str]:
    text = normalize_text(query)
    for phrase in sorted(QUERY_STOP_PHRASES, key=len, reverse=True):
        text = text.replace(phrase, "|")
    pieces = [piece.strip(" ：:，,。？?、") for piece in text.split("|")]
    terms = [piece for piece in pieces if len(piece) >= 2]
    token_terms = [token for token in tokenize(query) if len(token) >= 2]
    return list(dict.fromkeys(terms + token_terms))
