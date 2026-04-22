from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
DERIVED_DIR = DATA_DIR / "derived"
DOCUMENT_IMAGES_DIR = DERIVED_DIR / "document_images"
CHART_IMAGES_DIR = DERIVED_DIR / "chart_images"
PROCESSED_DIR = DATA_DIR / "processed"
TRAINING_DIR = DATA_DIR / "training"
REPORTS_DIR = DATA_DIR / "reports"
QA_VIZ_DIR = DATA_DIR / "qa_viz"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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


def deterministic_bucket(key: str, buckets: int = 100) -> int:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


def estimated_tokens(text: str) -> int:
    compact = re.sub(r"\s+", " ", text).strip()
    return max(1, len(compact) // 4)


def normalize_bbox(x: float, y: float, w: float, h: float, width: int, height: int) -> list[int]:
    xmin = int((x / width) * 1000)
    ymin = int((y / height) * 1000)
    xmax = int(((x + w) / width) * 1000)
    ymax = int(((y + h) / height) * 1000)
    return [
        max(0, min(1000, ymin)),
        max(0, min(1000, xmin)),
        max(0, min(1000, ymax)),
        max(0, min(1000, xmax)),
    ]


def relative_to_data(path: Path) -> str:
    return path.relative_to(DATA_DIR).as_posix()


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "item"


def parse_bbox_mentions(text: str) -> list[list[int]]:
    matches = re.findall(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", text)
    return [[int(part) for part in match] for match in matches]


def ensure_standard_dirs() -> None:
    for path in [
        DERIVED_DIR,
        DOCUMENT_IMAGES_DIR,
        CHART_IMAGES_DIR,
        PROCESSED_DIR,
        TRAINING_DIR,
        REPORTS_DIR,
        QA_VIZ_DIR,
    ]:
        ensure_dir(path)
