from __future__ import annotations

from pipeline_utils import PROCESSED_DIR, load_jsonl, write_jsonl

ASSET_FILE = PROCESSED_DIR / "asset_manifest.jsonl"
OUTPUT_FILE = PROCESSED_DIR / "llava_instruct.jsonl"


def describe_objects(object_counts: dict[str, int]) -> str:
    if not object_counts:
        return "The scene does not have reliable object labels, so the answer stays at the scene level."
    parts = [f"{label} x{count}" for label, count in object_counts.items()]
    return "Key objects include " + ", ".join(parts) + "."


def build_general_records(asset: dict) -> list[dict]:
    object_counts = asset.get("object_counts", {})
    top_label, top_count = next(iter(object_counts.items()), ("object", 0))
    description = (
        f"This image shows {asset['primary_caption']} "
        f"{describe_objects(object_counts)}"
    )
    count_answer = (
        f"The most frequent labeled category is {top_label}, with {top_count} annotated instance(s). "
        f"This answer is grounded in the local COCO annotation subset for this image."
    )
    return [
        {
            "id": f"{asset['asset_id']}_describe",
            "image": asset["image"],
            "task_type": "image_description",
            "asset_type": asset["asset_type"],
            "conversations": [
                {"from": "human", "value": "<image>\nPlease describe the scene, the main objects, and their relations."},
                {"from": "gpt", "value": description},
            ],
        },
        {
            "id": f"{asset['asset_id']}_count",
            "image": asset["image"],
            "task_type": "object_count_qa",
            "asset_type": asset["asset_type"],
            "conversations": [
                {"from": "human", "value": "<image>\nWhich object category appears most often, and how many instances are there?"},
                {"from": "gpt", "value": count_answer},
            ],
        },
    ]


def build_document_records(asset: dict) -> list[dict]:
    sections = asset["document_sections"]
    summary = " ".join(sections[:3])
    title = asset["title"]
    object_text = sections[2].replace("Candidate objects: ", "")
    return [
        {
            "id": f"{asset['asset_id']}_ocr_summary",
            "image": asset["image"],
            "task_type": "document_ocr_summary",
            "asset_type": asset["asset_type"],
            "conversations": [
                {"from": "human", "value": "<image>\nRead this document-style image and summarize the key information."},
                {"from": "gpt", "value": f"Title: {title}. Summary: {summary}"},
            ],
        },
        {
            "id": f"{asset['asset_id']}_doc_qa",
            "image": asset["image"],
            "task_type": "document_qa",
            "asset_type": asset["asset_type"],
            "conversations": [
                {"from": "human", "value": "<image>\nWhat title does this page use, and which objects are listed in the brief?"},
                {"from": "gpt", "value": f"The title is {title}. The listed objects are: {object_text}."},
            ],
        },
    ]


def build_chart_records(asset: dict) -> list[dict]:
    counts = asset.get("chart_counts", [])
    if counts:
        top = counts[0]
        second = counts[1] if len(counts) > 1 else None
        top_summary = f"The highest bar is {top['label']} with a count of {top['count']}."
        compare_summary = (
            f"The chart ranks {top['label']} first with {top['count']}, followed by {second['label']} with {second['count']}."
            if second
            else f"The chart only exposes one clear category, {top['label']}."
        )
    else:
        top_summary = "The chart does not contain reliable object bars."
        compare_summary = top_summary

    return [
        {
            "id": f"{asset['asset_id']}_chart_read",
            "image": asset["image"],
            "task_type": "chart_reading",
            "asset_type": asset["asset_type"],
            "conversations": [
                {"from": "human", "value": "<image>\nRead this chart and report the largest category."},
                {"from": "gpt", "value": top_summary},
            ],
        },
        {
            "id": f"{asset['asset_id']}_chart_compare",
            "image": asset["image"],
            "task_type": "chart_comparison",
            "asset_type": asset["asset_type"],
            "conversations": [
                {"from": "human", "value": "<image>\nCompare the top object categories shown in the chart."},
                {"from": "gpt", "value": compare_summary},
            ],
        },
    ]


def main() -> None:
    assets = load_jsonl(ASSET_FILE)
    dataset: list[dict] = []

    for asset in assets:
        if asset["asset_type"] == "general_image":
            dataset.extend(build_general_records(asset))
        elif asset["asset_type"] == "document_image":
            dataset.extend(build_document_records(asset))
        elif asset["asset_type"] == "chart_image":
            dataset.extend(build_chart_records(asset))

    write_jsonl(dataset, OUTPUT_FILE)
    print("✅ LLaVA 基础指令数据生成完成。")
    print({"num_records": len(dataset)})


if __name__ == "__main__":
    main()
