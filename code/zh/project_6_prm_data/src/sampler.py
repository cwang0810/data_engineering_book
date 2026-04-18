from __future__ import annotations

from collections import Counter

from pipeline_utils import (
    GSM8K_FILE,
    MBPP_FILE,
    PROCESSED_DIR,
    ensure_standard_dirs,
    extract_final_answer,
    load_jsonl,
    split_reasoning_steps,
    write_json,
    write_jsonl,
)

SEED_FILE = PROCESSED_DIR / "seed_pool.jsonl"
TASK_SPEC_FILE = PROCESSED_DIR / "task_spec.json"

NUM_MATH = 24
NUM_CODE = 12


def infer_math_topic(question: str) -> str:
    text = question.lower()
    if "%" in text or "percent" in text or "tax" in text:
        return "percentages"
    if "mile" in text or "hour" in text or "minute" in text or "speed" in text:
        return "rates_and_motion"
    if "fraction" in text or "/" in text:
        return "fractions"
    return "arithmetic"


def infer_code_topic(prompt: str) -> str:
    text = prompt.lower()
    if "string" in text or "character" in text:
        return "string_manipulation"
    if "list" in text or "array" in text:
        return "list_processing"
    if "search" in text or "sort" in text:
        return "search_sort"
    return "function_design"


def main() -> None:
    ensure_standard_dirs()
    gsm8k = load_jsonl(GSM8K_FILE)
    mbpp = load_jsonl(MBPP_FILE)

    seeds: list[dict] = []
    for index, record in enumerate(gsm8k):
        final_answer = extract_final_answer(record["answer"])
        steps = split_reasoning_steps(record["answer"])
        if not final_answer or len(steps) < 2:
            continue
        seeds.append(
            {
                "seed_id": f"math_{index}",
                "domain": "math",
                "topic": infer_math_topic(record["question"]),
                "question": record["question"],
                "reference_steps": steps,
                "final_answer": final_answer,
                "source_dataset": "gsm8k_train",
            }
        )
        if sum(seed["domain"] == "math" for seed in seeds) >= NUM_MATH:
            break

    code_count = 0
    for record in mbpp:
        if len(record.get("test_list", [])) < 2:
            continue
        seeds.append(
            {
                "seed_id": f"code_{record['task_id']}",
                "domain": "code",
                "topic": infer_code_topic(record["text"]),
                "question": record["text"],
                "reference_code": record["code"],
                "test_setup_code": record.get("test_setup_code", ""),
                "test_list": record.get("test_list", []),
                "challenge_test_list": record.get("challenge_test_list", []),
                "source_dataset": "mbpp_train",
            }
        )
        code_count += 1
        if code_count >= NUM_CODE:
            break

    task_spec = {
        "project_goal": "Build a reproducible CoT + PRM mini dataset for math and code reasoning.",
        "seed_count": len(seeds),
        "domain_distribution": dict(Counter(seed["domain"] for seed in seeds)),
        "topic_distribution": dict(Counter(seed["topic"] for seed in seeds)),
        "trace_targets": {
            "positive": "correct reasoning trace",
            "negative": "corrupted or wrong reasoning trace",
            "repair": "wrong step followed by correction",
        },
        "validation_targets": [
            "final_answer_match",
            "step_level_quality_labels",
            "code_execution_and_unit_tests",
        ],
    }

    write_jsonl(seeds, SEED_FILE)
    write_json(task_spec, TASK_SPEC_FILE)
    print("✅ P6 种子池与任务规格生成完成。")
    print(task_spec)


if __name__ == "__main__":
    main()
