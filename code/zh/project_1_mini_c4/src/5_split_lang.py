from __future__ import annotations

import json
import os

from pipeline_utils import normalize_text

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "lid.176.ftz")

INPUT_FILE = os.path.join(DATA_DIR, "deduplicated_data.jsonl")
OUTPUT_FILES = {
    "en": os.path.join(DATA_DIR, "data_en.jsonl"),
    "zh": os.path.join(DATA_DIR, "data_zh.jsonl"),
    "others": os.path.join(DATA_DIR, "data_others.jsonl"),
}

MIN_TEXT_CHARS = 40


def smoke_detect_lang(text: str) -> tuple[str, float]:
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    latin = sum(1 for char in text if char.isascii() and char.isalpha())
    if cjk >= latin:
        return "zh", 0.99
    return "en", 0.99


def main() -> None:
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        return

    smoke_mode = os.environ.get("PROJECT_SMOKE") == "1"
    model = None
    if not smoke_mode and not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到 FastText 模型: {MODEL_PATH}")
        return

    if not smoke_mode:
        import fasttext
        model = fasttext.load_model(MODEL_PATH)

    stats = {"total": 0, "en": 0, "zh": 0, "others": 0, "skipped": 0}

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_FILES["en"], "w", encoding="utf-8") as f_en, \
         open(OUTPUT_FILES["zh"], "w", encoding="utf-8") as f_zh, \
         open(OUTPUT_FILES["others"], "w", encoding="utf-8") as f_others:

        writers = {"en": f_en, "zh": f_zh, "others": f_others}

        for line in f_in:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                stats["skipped"] += 1
                continue

            text = normalize_text(data.get("text", ""))
            if len(text) < MIN_TEXT_CHARS:
                stats["skipped"] += 1
                continue

            if smoke_mode:
                detected_lang, confidence = smoke_detect_lang(text)
            else:
                labels, scores = model.predict(text.replace("\n", " "), k=1)
                detected_lang = labels[0].replace("__label__", "")
                confidence = float(scores[0])

            data["text"] = text
            data["detected_lang"] = detected_lang
            data["lang_confidence"] = round(confidence, 6)

            if detected_lang == "en":
                bucket = "en"
            elif detected_lang == "zh":
                bucket = "zh"
            else:
                bucket = "others"

            writers[bucket].write(json.dumps(data, ensure_ascii=False) + "\n")

            stats["total"] += 1
            stats[bucket] += 1

    print("语言切分完成！")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
