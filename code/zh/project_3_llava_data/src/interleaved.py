from __future__ import annotations

from pipeline_utils import PROCESSED_DIR, load_jsonl, write_jsonl

ASSET_FILE = PROCESSED_DIR / "asset_manifest.jsonl"
OUTPUT_FILE = PROCESSED_DIR / "llava_interleaved.jsonl"


def compare_objects(left: dict, right: dict) -> str:
    left_caption = left["primary_caption"]
    right_caption = right["primary_caption"]
    left_objects = ", ".join(f"{label} x{count}" for label, count in left.get("object_counts", {}).items()) or "no strong labels"
    right_objects = ", ".join(f"{label} x{count}" for label, count in right.get("object_counts", {}).items()) or "no strong labels"
    return (
        f"Image 1: {left_caption} Key objects: {left_objects}. "
        f"Image 2: {right_caption} Key objects: {right_objects}. "
        "Both answers are grounded in the local captions and object annotations."
    )


def main() -> None:
    assets = [asset for asset in load_jsonl(ASSET_FILE) if asset["asset_type"] == "general_image"]
    assets = sorted(assets, key=lambda item: item["image"])
    records: list[dict] = []

    for index in range(0, len(assets) - 1, 2):
        left = assets[index]
        right = assets[index + 1]
        records.append(
            {
                "id": f"interleaved_{left['image_id']}_{right['image_id']}",
                "image": [left["image"], right["image"]],
                "task_type": "multi_image_comparison",
                "asset_type": "interleaved_pair",
                "conversations": [
                    {"from": "human", "value": "Image 1: <image>\nImage 2: <image>\nCompare the two scenes and mention shared or distinct objects."},
                    {"from": "gpt", "value": compare_objects(left, right)},
                ],
            }
        )

    write_jsonl(records, OUTPUT_FILE)
    print("✅ 多图交错数据生成完成。")
    print({"num_records": len(records)})


if __name__ == "__main__":
    main()
