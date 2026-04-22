from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ARTICLE_RE = re.compile(r"^(第[0-9零一二三四五六七八九十百千]+条)")


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    return project_root() / "data"


def processed_dir() -> Path:
    return data_dir() / "processed"


def training_dir() -> Path:
    return data_dir() / "training"


def reports_dir() -> Path:
    return data_dir() / "reports"


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def law_name_from_source(filename: str) -> str:
    return filename.replace(".pdf", "")


def extract_article_no(content: str) -> str:
    match = ARTICLE_RE.match(content.strip())
    return match.group(1) if match else "未编号条文"


def build_seed_id(law_name: str, article_no: str, content: str) -> str:
    short_hash = sha1_text(f"{law_name}|{article_no}|{content}")[:12]
    return f"{law_name[:8]}_{article_no[:8]}_{short_hash}"


def estimated_tokens(text: str) -> int:
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_words = len(re.findall(r"[A-Za-z0-9_]+", text))
    return cjk_chars + latin_words


def deterministic_bucket(text: str, modulo: int = 100) -> int:
    return int(sha1_text(text)[:8], 16) % modulo


def trim_summary(text: str, max_len: int = 120) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1] + "…"
