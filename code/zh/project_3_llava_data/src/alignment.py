from __future__ import annotations

from pipeline_utils import PROCESSED_DIR, load_jsonl, normalize_bbox, write_jsonl

ASSET_FILE = PROCESSED_DIR / "asset_manifest.jsonl"
OUTPUT_FILE = PROCESSED_DIR / "llava_alignment.jsonl"


def main() -> None:
    assets = load_jsonl(ASSET_FILE)
    records: list[dict] = []

    for asset in assets:
        if asset["asset_type"] != "general_image":
            continue

        width = asset["width"]
        height = asset["height"]
        for index, annotation in enumerate(asset.get("top_annotations", []), start=1):
            bbox = normalize_bbox(*annotation["bbox_xywh"], width, height)
            label = annotation["label"]
            records.append(
                {
                    "id": f"{asset['asset_id']}_bbox_{index}",
                    "image": asset["image"],
                    "task_type": "region_grounding",
                    "asset_type": asset["asset_type"],
                    "bbox": bbox,
                    "label": label,
                    "conversations": [
                        {"from": "human", "value": f"<image>\nLocate the {label} in the image and return its bounding box."},
                        {
                            "from": "gpt",
                            "value": f"The {label} is located at [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}].",
                        },
                    ],
                }
            )

    write_jsonl(records, OUTPUT_FILE)
    print("✅ 区域对齐数据生成完成。")
    print({"num_records": len(records)})


if __name__ == "__main__":
    main()
