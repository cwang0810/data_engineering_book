from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TRAINING_DIR = DATA_DIR / "training"
REPORTS_DIR = DATA_DIR / "reports"
BOOKS_DIR = ROOT_DIR / "books"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_standard_dirs() -> None:
    for path in [PROCESSED_DIR, TRAINING_DIR, REPORTS_DIR, BOOKS_DIR]:
        ensure_dir(path)


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def deterministic_bucket(key: str, buckets: int = 100) -> int:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


def estimated_tokens(text: str) -> int:
    compact = re.sub(r"\s+", " ", text).strip()
    return max(1, len(compact) // 4)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_gsm8k_solution(answer: str) -> str:
    text = re.sub(r"<<([^>]+)>>", r"\1", answer)
    text = text.replace("####", "Final answer:")
    return normalize_text(text)


def extract_final_answer(answer: str) -> str | None:
    match = re.search(r"####\s*(.+)$", answer, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None
