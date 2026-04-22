from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pipeline_utils import estimate_token_count, extract_domain

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TRAINING_DIR = DATA_DIR / "training"
REPORT_DIR = DATA_DIR / "reports"

FILES = {
    "extracted": PROCESSED_DIR / "extracted_data.jsonl",
    "clean": PROCESSED_DIR / "clean_data.jsonl",
    "deduplicated": PROCESSED_DIR / "deduplicated_data.jsonl",
    "lang_en": PROCESSED_DIR / "data_en.jsonl",
    "lang_zh": PROCESSED_DIR / "data_zh.jsonl",
    "final": PROCESSED_DIR / "final_data.jsonl",
}

QUALITY_STATS_FILE = PROCESSED_DIR / "quality_filter_stats.json"
TRAINING_MANIFEST_FILE = TRAINING_DIR / "training_manifest.json"
METRICS_JSON = REPORT_DIR / "p1_metrics.json"
REPORT_MD = REPORT_DIR / "p1_report.md"


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return round(path.stat().st_size / (1024 * 1024), 2)


def directory_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total_bytes = 0
    for child in path.rglob("*"):
        if child.is_file():
            total_bytes += child.stat().st_size
    return round(total_bytes / (1024 ** 3), 2)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_final_dataset(path: Path) -> dict:
    language_counter = Counter()
    domain_counter = Counter()
    doc_count = 0
    char_count = 0
    token_count = 0

    if not path.exists():
        return {
            "doc_count": 0,
            "char_count": 0,
            "estimated_tokens": 0,
            "avg_chars_per_doc": 0.0,
            "avg_tokens_per_doc": 0.0,
            "languages": {},
            "top_domains": [],
        }

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = item.get("text", "")
            doc_count += 1
            char_count += len(text)
            token_count += estimate_token_count(text)
            language_counter[item.get("lang", "unknown")] += 1
            domain_counter[extract_domain(item.get("url", ""))] += 1

    return {
        "doc_count": doc_count,
        "char_count": char_count,
        "estimated_tokens": token_count,
        "avg_chars_per_doc": round(char_count / doc_count, 2) if doc_count else 0.0,
        "avg_tokens_per_doc": round(token_count / doc_count, 2) if doc_count else 0.0,
        "languages": dict(language_counter),
        "top_domains": domain_counter.most_common(10),
    }


def build_metrics() -> dict:
    stage_counts = {name: count_lines(path) for name, path in FILES.items()}
    stage_sizes_mb = {name: file_size_mb(path) for name, path in FILES.items()}
    final_summary = summarize_final_dataset(FILES["final"])
    quality_stats = load_json(QUALITY_STATS_FILE)
    training_manifest = load_json(TRAINING_MANIFEST_FILE)

    extracted_count = stage_counts.get("extracted", 0) or 1
    clean_count = stage_counts.get("clean", 0)
    dedup_count = stage_counts.get("deduplicated", 0)
    final_count = stage_counts.get("final", 0)

    retention = {
        "clean_over_extracted": round(clean_count / extracted_count, 4),
        "dedup_over_extracted": round(dedup_count / extracted_count, 4),
        "final_over_extracted": round(final_count / extracted_count, 4),
    }

    raw_disk_gb = directory_size_gb(DATA_DIR / "raw")
    processed_disk_gb = directory_size_gb(PROCESSED_DIR)
    model_disk_gb = directory_size_gb(PROJECT_ROOT / "models")
    total_disk_gb = round(raw_disk_gb + processed_disk_gb + model_disk_gb, 2)
    recommended_free_disk_gb = round(total_disk_gb * 1.5, 2)
    monthly_storage_cost_usd = round(total_disk_gb * 0.023, 2)

    return {
        "stage_counts": stage_counts,
        "stage_sizes_mb": stage_sizes_mb,
        "retention": retention,
        "final_summary": final_summary,
        "quality_stats": quality_stats,
        "training_manifest": training_manifest,
        "hardware_boundary": {
            "raw_disk_gb": raw_disk_gb,
            "processed_disk_gb": processed_disk_gb,
            "model_disk_gb": model_disk_gb,
            "total_disk_gb": total_disk_gb,
            "recommended_free_disk_gb": recommended_free_disk_gb,
            "execution_mode": "single-node CPU preprocessing; Ray for MinHash dedup; KenLM for English quality scoring",
        },
        "cost_analysis": {
            "monthly_storage_cost_usd_estimate": monthly_storage_cost_usd,
            "download_bandwidth_gb": raw_disk_gb,
            "notes": [
                "Common Crawl download is free, but local bandwidth and storage are the dominant dataset-building costs.",
                "Model footprint is larger than processed data footprint, so quality-filter models should be shared across experiments.",
                "The current mini scope is suitable for laptop or single-server CPU preprocessing.",
            ],
            "optimization_space": [
                "Replace full-file reads in dedup with streaming or chunked indexing for larger crawls.",
                "Add domain allow/block lists before language split to reduce wasted parsing.",
                "Train or import a Chinese LM for zh quality scoring to reduce mixed-language leakage.",
                "Persist intermediate metrics per stage to measure runtime, not just data volume.",
            ],
        },
    }


def render_report(metrics: dict) -> str:
    stage_counts = metrics["stage_counts"]
    retention = metrics["retention"]
    final_summary = metrics["final_summary"]
    hardware = metrics["hardware_boundary"]
    cost = metrics["cost_analysis"]
    training = metrics.get("training_manifest", {})

    top_domains_lines = "\n".join(
        f"- {domain}: {count}" for domain, count in final_summary.get("top_domains", [])
    ) or "- None"

    return f"""# P1 Mini-C4 Delivery Report

## 1. Project Goals And Data Scope

- Goal: build a reproducible Mini-C4 style preprocessing pipeline on a small Common Crawl slice.
- Scope: single WARC file, CPU-only local preprocessing, English/Chinese split, final training-ready JSONL export.
- Hardware boundary: raw {hardware['raw_disk_gb']} GB, processed {hardware['processed_disk_gb']} GB, models {hardware['model_disk_gb']} GB, recommended free disk >= {hardware['recommended_free_disk_gb']} GB.

## 2. Data Retrieval And Parsing

- Extracted records: {stage_counts.get('extracted', 0)}
- Cleaned records: {stage_counts.get('clean', 0)}
- Deduplicated records: {stage_counts.get('deduplicated', 0)}
- Final records: {stage_counts.get('final', 0)}
- Pipeline path: download -> WARC parse -> cleaning -> MinHash dedup -> language split -> quality filtering.

## 3. Cleaning, Deduplication And Quality Filtering

- Clean retention: {retention['clean_over_extracted']:.2%}
- Dedup retention: {retention['dedup_over_extracted']:.2%}
- Final retention: {retention['final_over_extracted']:.2%}
- Final language distribution: {json.dumps(final_summary.get('languages', {}), ensure_ascii=False)}
- Top final domains:
{top_domains_lines}

## 4. Serialization And Training Preparation

- Serialized dataset records: {training.get('num_records', 0)}
- Train records: {training.get('num_train_records', 0)}
- Val records: {training.get('num_val_records', 0)}
- Smoke test records: {training.get('num_smoke_test_records', 0)}
- Estimated tokens (train): {training.get('estimated_tokens_train', 0)}
- Outputs: `serialized_dataset.jsonl`, `train.jsonl`, `val.jsonl`, shard files, `smoke_test.jsonl`, and `training_manifest.json`.

## 5. Evaluation And Cost Analysis

- Avg chars/doc: {final_summary.get('avg_chars_per_doc', 0)}
- Avg estimated tokens/doc: {final_summary.get('avg_tokens_per_doc', 0)}
- Estimated monthly storage cost at $0.023/GB-month: ${cost['monthly_storage_cost_usd_estimate']}
- Dominant cost structure: disk footprint, download bandwidth, and CPU-based filtering/models.

## 6. Extension Directions

- Scale from one WARC shard to multiple shards with chunked dedup and per-stage runtime logs.
- Add stronger domain filtering before parse to reduce obvious spam/adult/menu pages.
- Add Chinese LM scoring and tokenizer-aware packaging for multilingual pretraining.
- Connect downstream chapters by reusing `data/training/` artifacts as tokenizer or pretraining inputs.
"""


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = build_metrics()

    with METRICS_JSON.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    report = render_report(metrics)
    with REPORT_MD.open("w", encoding="utf-8") as f:
        f.write(report)

    print("Dataset evaluation complete.")
    print(report)


if __name__ == "__main__":
    main()
