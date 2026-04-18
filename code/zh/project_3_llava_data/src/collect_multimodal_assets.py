from __future__ import annotations

import textwrap
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from pipeline_utils import (
    ANNOTATIONS_DIR,
    CHART_IMAGES_DIR,
    DOCUMENT_IMAGES_DIR,
    IMAGES_DIR,
    PROCESSED_DIR,
    ensure_standard_dirs,
    load_json,
    relative_to_data,
    slugify,
    write_json,
    write_jsonl,
)

OUTPUT_FILE = PROCESSED_DIR / "asset_manifest.jsonl"
SUMMARY_FILE = PROCESSED_DIR / "asset_collection_summary.json"


def wrap_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        text_width = draw.textbbox((0, 0), candidate, font=font)[2]
        if text_width <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def render_document_card(output_path: Path, title: str, sections: list[str]) -> None:
    image = Image.new("RGB", (1200, 900), color=(248, 247, 242))
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()

    draw.rectangle((36, 36, 1164, 864), outline=(35, 43, 56), width=3)
    draw.text((72, 72), title, fill=(27, 38, 59), font=title_font)

    y = 130
    max_width = 1040
    for section in sections:
        for line in wrap_lines(draw, section, body_font, max_width):
            draw.text((72, y), line, fill=(55, 65, 81), font=body_font)
            y += 26
        y += 12

    image.save(output_path)


def render_chart_card(output_path: Path, title: str, counts: list[tuple[str, int]]) -> None:
    image = Image.new("RGB", (1200, 800), color=(252, 252, 251))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((60, 40), title, fill=(24, 39, 75), font=font)

    if not counts:
        draw.text((60, 120), "No object counts available.", fill=(95, 99, 104), font=font)
        image.save(output_path)
        return

    max_count = max(count for _, count in counts)
    base_x = 240
    y = 120
    for idx, (label, count) in enumerate(counts):
        color = (75 + idx * 18, 130 + idx * 8, 180 - idx * 10)
        bar_width = int(700 * (count / max_count))
        draw.text((60, y + 8), label, fill=(55, 65, 81), font=font)
        draw.rectangle((base_x, y, base_x + bar_width, y + 42), fill=color)
        draw.text((base_x + bar_width + 12, y + 8), str(count), fill=(55, 65, 81), font=font)
        y += 90

    image.save(output_path)


def main() -> None:
    ensure_standard_dirs()

    instances = load_json(ANNOTATIONS_DIR / "instances_val2017.json")
    captions = load_json(ANNOTATIONS_DIR / "captions_val2017.json")

    categories = {item["id"]: item["name"] for item in instances["categories"]}
    images_by_id = {item["id"]: item for item in instances["images"]}

    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in instances["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    captions_by_image: dict[int, list[str]] = defaultdict(list)
    for ann in captions["annotations"]:
        captions_by_image[ann["image_id"]].append(ann["caption"])

    records: list[dict] = []
    image_paths = sorted(IMAGES_DIR.glob("*.jpg"))

    for image_path in tqdm(image_paths, desc="Collecting multimodal assets"):
        image_id = int(image_path.stem)
        image_info = images_by_id[image_id]
        width = image_info["width"]
        height = image_info["height"]
        image_captions = captions_by_image.get(image_id, [])
        primary_caption = image_captions[0] if image_captions else "A natural image from the COCO validation split."

        anns = anns_by_image.get(image_id, [])
        object_counts = Counter(categories[ann["category_id"]] for ann in anns)
        top_objects = object_counts.most_common(5)

        general_record = {
            "asset_id": f"general_{image_path.stem}",
            "asset_type": "general_image",
            "image": relative_to_data(image_path),
            "source_image": image_path.name,
            "image_id": image_id,
            "width": width,
            "height": height,
            "captions": image_captions[:3],
            "primary_caption": primary_caption,
            "object_counts": dict(top_objects),
            "top_annotations": [
                {
                    "label": categories[ann["category_id"]],
                    "bbox_xywh": ann["bbox"],
                    "area": ann["area"],
                }
                for ann in sorted(anns, key=lambda item: item["area"], reverse=True)[:3]
            ],
        }
        records.append(general_record)

        doc_path = DOCUMENT_IMAGES_DIR / f"{image_path.stem}_document.png"
        sections = [
            f"Source image id: {image_id}",
            f"Primary caption: {primary_caption}",
            "Candidate objects: " + ", ".join(f"{label} x{count}" for label, count in top_objects) if top_objects else "Candidate objects: none",
            "Use case: OCR-like reading, caption rewriting, and evidence-based QA.",
        ]
        render_document_card(doc_path, f"P3 Vision Brief {image_path.stem}", sections)
        records.append(
            {
                "asset_id": f"document_{image_path.stem}",
                "asset_type": "document_image",
                "image": relative_to_data(doc_path),
                "source_image": image_path.name,
                "image_id": image_id,
                "title": f"P3 Vision Brief {image_path.stem}",
                "document_sections": sections,
                "primary_caption": primary_caption,
                "object_counts": dict(top_objects),
            }
        )

        chart_path = CHART_IMAGES_DIR / f"{image_path.stem}_chart.png"
        render_chart_card(chart_path, f"Object Count Summary {image_path.stem}", top_objects)
        records.append(
            {
                "asset_id": f"chart_{image_path.stem}",
                "asset_type": "chart_image",
                "image": relative_to_data(chart_path),
                "source_image": image_path.name,
                "image_id": image_id,
                "chart_title": f"Object Count Summary {image_path.stem}",
                "chart_counts": [{"label": label, "count": count} for label, count in top_objects],
                "primary_caption": primary_caption,
            }
        )

    summary = {
        "num_source_images": len(image_paths),
        "num_assets_total": len(records),
        "asset_type_distribution": dict(Counter(record["asset_type"] for record in records)),
    }

    write_jsonl(records, OUTPUT_FILE)
    write_json(summary, SUMMARY_FILE)
    print("✅ 多模态资产收集完成。")
    print(summary)


if __name__ == "__main__":
    main()
