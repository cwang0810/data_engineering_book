from __future__ import annotations

import json
import os

from tqdm import tqdm

from pipeline_utils import (
    character_ratio,
    duplicate_line_ratio,
    estimate_token_count,
    median_line_length,
    normalize_text,
    script_counts,
    sha1_text,
    short_line_ratio,
    split_nonempty_lines,
)

try:
    import kenlm
except ImportError:
    kenlm = None


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

INPUT_FILES = {
    "en": os.path.join(DATA_DIR, "data_en.jsonl"),
    "zh": os.path.join(DATA_DIR, "data_zh.jsonl"),
}
OUTPUT_FILES = {
    "en": os.path.join(DATA_DIR, "final_data_en.jsonl"),
    "zh": os.path.join(DATA_DIR, "final_data_zh.jsonl"),
}
MERGED_OUTPUT_FILE = os.path.join(DATA_DIR, "final_data.jsonl")
STATS_FILE = os.path.join(DATA_DIR, "quality_filter_stats.json")
MODEL_PATH = os.path.join(MODEL_DIR, "en.arpa.bin")

PERPLEXITY_THRESHOLD = -6.0
MIN_CHARS = {"en": 160, "zh": 120}

NAVIGATION_KEYWORDS = {
    "home", "about", "contact", "privacy", "terms", "copyright",
    "login", "register", "search", "menu", "help", "forum", "news",
    "首页", "登录", "注册", "联系我们", "关于我们", "隐私", "版权", "返回顶部",
    "目录", "搜索", "帮助", "校历", "课程表", "Contact Us",
}
ADULT_SPAM_PHRASES = [
    "自拍偷拍", "裸聊", "直播平台", "无码", "成人", "麻豆", "约炮", "台灣uu",
    "真愛旅舍", "大尺度", "美女直播", "性色", "性聊", "91视频", "色情",
    "casino", "sportsbook", "poker", "xxx", "porn", "escort",
]


def is_navigation_line(line: str) -> bool:
    lower = line.lower()
    if len(line) <= 32 and any(keyword.lower() in lower for keyword in NAVIGATION_KEYWORDS):
        return True
    if len(line) <= 48 and (line.count("|") >= 2 or line.count("/") >= 4):
        return True
    if "all rights reserved" in lower or "copyright" in lower:
        return True
    return False


def reject_reason(text: str, lang: str, lm_model) -> tuple[str | None, dict]:
    normalized = normalize_text(text)
    if not normalized:
        return "empty", {}

    lines = split_nonempty_lines(normalized)
    counts = script_counts(normalized)
    estimated_tokens = estimate_token_count(normalized)
    nav_ratio = (
        sum(1 for line in lines if is_navigation_line(line)) / len(lines) if lines else 0.0
    )
    duplicate_ratio = duplicate_line_ratio(lines)
    short_ratio = short_line_ratio(lines)
    median_len = median_line_length(lines)
    digit_ratio = character_ratio(normalized, "digits")

    metadata = {
        "estimated_tokens": estimated_tokens,
        "script_counts": counts,
        "line_count": len(lines),
        "duplicate_line_ratio": round(duplicate_ratio, 4),
        "navigation_line_ratio": round(nav_ratio, 4),
    }

    if len(normalized) < MIN_CHARS[lang]:
        return "too_short", metadata
    if estimated_tokens < 40:
        return "too_few_tokens", metadata
    if duplicate_ratio > 0.2:
        return "duplicate_lines", metadata
    if len(lines) >= 12 and short_ratio > 0.75 and median_len < 28:
        return "directory_like", metadata
    if nav_ratio > 0.35:
        return "navigation_heavy", metadata
    if digit_ratio > 0.2:
        return "digit_heavy", metadata

    lowered = normalized.lower()
    adult_hits = sum(1 for phrase in ADULT_SPAM_PHRASES if phrase.lower() in lowered)
    if adult_hits >= 2:
        return "adult_or_spam", metadata

    if lang == "en":
        if counts["latin"] < 120:
            return "insufficient_english", metadata
        if counts["cjk"] > max(40, counts["latin"] * 0.1):
            return "mixed_language_en", metadata
        if lm_model is not None:
            words = max(1, len(normalized.split()))
            normalized_score = lm_model.score(normalized) / words
            metadata["perplexity_score"] = round(normalized_score, 6)
            if normalized_score <= PERPLEXITY_THRESHOLD:
                return "low_kenlm_score", metadata
    elif lang == "zh":
        if counts["cjk"] < 80:
            return "insufficient_chinese", metadata
        if counts["latin"] > counts["cjk"]:
            return "mixed_language_zh", metadata

    return None, metadata


def filter_file(lang: str, input_path: str, output_path: str, lm_model) -> tuple[dict, list[dict]]:
    stats = {
        "total": 0,
        "kept": 0,
        "dropped": 0,
        "reasons": {},
    }
    kept_records = []
    seen_hashes = set()

    if not os.path.exists(input_path):
        return stats, kept_records

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in, desc=f"Filtering {lang}"):
            stats["total"] += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                stats["dropped"] += 1
                stats["reasons"]["json_error"] = stats["reasons"].get("json_error", 0) + 1
                continue

            text = normalize_text(item.get("text", ""))
            reason, metadata = reject_reason(text, lang, lm_model)
            if reason is not None:
                stats["dropped"] += 1
                stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
                continue

            text_sha1 = sha1_text(text)
            if text_sha1 in seen_hashes:
                stats["dropped"] += 1
                stats["reasons"]["exact_duplicate"] = stats["reasons"].get("exact_duplicate", 0) + 1
                continue
            seen_hashes.add(text_sha1)

            item["text"] = text
            item["lang"] = lang
            item["text_sha1"] = text_sha1
            for key, value in metadata.items():
                item[key] = value

            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept_records.append(item)
            stats["kept"] += 1

    return stats, kept_records


def main() -> None:
    lm_model = None
    if kenlm is not None and os.path.exists(MODEL_PATH):
        print(f"🚀 加载 KenLM 模型: {MODEL_PATH}")
        lm_model = kenlm.Model(MODEL_PATH)

    all_stats = {}
    merged_records = []

    for lang in ("en", "zh"):
        stats, kept_records = filter_file(
            lang=lang,
            input_path=INPUT_FILES[lang],
            output_path=OUTPUT_FILES[lang],
            lm_model=lm_model if lang == "en" else None,
        )
        all_stats[lang] = stats
        merged_records.extend(kept_records)

    with open(MERGED_OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in merged_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    print("质量过滤完成！")
    print(json.dumps(all_stats, ensure_ascii=False, indent=2))
    print(f"💾 合并输出: {MERGED_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
