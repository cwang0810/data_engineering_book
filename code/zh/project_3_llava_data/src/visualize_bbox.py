from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pipeline_utils import DATA_DIR, IMAGES_DIR, PROCESSED_DIR, QA_VIZ_DIR, ensure_dir, load_jsonl, write_jsonl

INPUT_FILE = PROCESSED_DIR / "llava_alignment.jsonl"
MANIFEST_FILE = PROCESSED_DIR / "qa_visual_audit.jsonl"


def denormalize_bbox(bbox: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    ymin, xmin, ymax, xmax = bbox
    return (
        int(xmin / 1000 * width),
        int(ymin / 1000 * height),
        int(xmax / 1000 * width),
        int(ymax / 1000 * height),
    )


def main() -> None:
    ensure_dir(QA_VIZ_DIR)
    records = load_jsonl(INPUT_FILE)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["image"]].append(record)

    manifest: list[dict] = []
    font = ImageFont.load_default()

    for image_rel, image_records in grouped.items():
        image_path = DATA_DIR / image_rel
        if not image_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for record in image_records:
            x1, y1, x2, y2 = denormalize_bbox(record["bbox"], width, height)
            draw.rectangle((x1, y1, x2, y2), outline=(255, 64, 64), width=3)
            draw.text((x1 + 4, max(0, y1 - 14)), record["label"], fill=(255, 64, 64), font=font)

        output_path = QA_VIZ_DIR / f"viz_{Path(image_rel).name}"
        image.save(output_path)
        manifest.append(
            {
                "image": image_rel,
                "output_image": output_path.relative_to(DATA_DIR).as_posix(),
                "num_boxes": len(image_records),
                "labels": [record["label"] for record in image_records],
            }
        )

    write_jsonl(manifest, MANIFEST_FILE)
    print("✅ 可视化抽检图生成完成。")
    print({"num_visualizations": len(manifest)})


if __name__ == "__main__":
    main()
