from __future__ import annotations

from collections import Counter
from pathlib import Path

from pipeline_utils import (
    DATA_DIR,
    PROCESSED_DIR,
    clean_gsm8k_solution,
    ensure_standard_dirs,
    extract_final_answer,
    load_jsonl,
    write_json,
    write_jsonl,
)

GSM8K_FILE = DATA_DIR / "gsm8k_train.jsonl"
MBPP_FILE = DATA_DIR / "mbpp_train.jsonl"
SEED_FILE = PROCESSED_DIR / "seed_pool.jsonl"
PLAN_FILE = PROCESSED_DIR / "chapter_plan.json"

MATH_TARGET = 30
CODE_TARGET = 18


def infer_math_topic(question: str) -> str:
    text = question.lower()
    if "%" in text or "percent" in text or "tax" in text:
        return "percentages_and_rates"
    if "mile" in text or "hour" in text or "minute" in text or "speed" in text:
        return "rates_and_motion"
    if "area" in text or "rectangle" in text or "circle" in text or "perimeter" in text:
        return "geometry_and_measurement"
    if "fraction" in text or "/" in text:
        return "fractions_and_proportions"
    return "arithmetic_word_problems"


def infer_code_topic(prompt: str) -> str:
    text = prompt.lower()
    if "string" in text or "character" in text:
        return "string_algorithms"
    if "list" in text or "array" in text:
        return "lists_and_iteration"
    if "tree" in text or "graph" in text:
        return "graphs_and_trees"
    if "sort" in text or "search" in text:
        return "search_and_sort"
    return "function_design"


def infer_difficulty(text: str, domain: str) -> str:
    length = len(text.split())
    if domain == "math":
        if length < 18:
            return "intro"
        if length < 32:
            return "core"
        return "advanced"
    if length < 10:
        return "core"
    if length < 16:
        return "advanced"
    return "extension"


def build_chapter_templates() -> dict:
    return {
        "math": {
            "audience": "middle_school_to_early_high_school",
            "formats": ["concept_note", "worked_example", "checkpoint_exercise", "verification_snippet"],
            "chapters": [
                "Arithmetic and quantity tracking",
                "Percentages, taxes, and residual values",
                "Rates, time, and motion",
                "Fractions, ratios, and proportional reasoning",
            ],
        },
        "code": {
            "audience": "introductory_python_learners",
            "formats": ["concept_note", "annotated_solution", "unit_test_block", "debugging_tip"],
            "chapters": [
                "Function design and decomposition",
                "Strings and list processing",
                "Searching, sorting, and dynamic patterns",
                "Testing and debugging with assertions",
            ],
        },
    }


def main() -> None:
    ensure_standard_dirs()
    gsm8k_records = load_jsonl(GSM8K_FILE)
    mbpp_records = load_jsonl(MBPP_FILE)

    math_seeds: list[dict] = []
    for index, record in enumerate(gsm8k_records):
        final_answer = extract_final_answer(record["answer"])
        if not final_answer:
            continue
        math_seeds.append(
            {
                "id": f"math_{index}",
                "domain": "math",
                "topic": infer_math_topic(record["question"]),
                "difficulty": infer_difficulty(record["question"], "math"),
                "source_dataset": "gsm8k_train",
                "seed_question": record["question"],
                "source_solution": clean_gsm8k_solution(record["answer"]),
                "expected_answer": final_answer,
            }
        )
        if len(math_seeds) >= MATH_TARGET:
            break

    code_seeds: list[dict] = []
    for index, record in enumerate(mbpp_records):
        code_seeds.append(
            {
                "id": f"code_{record['task_id']}",
                "domain": "code",
                "topic": infer_code_topic(record["text"]),
                "difficulty": infer_difficulty(record["text"], "code"),
                "source_dataset": "mbpp_train",
                "seed_question": record["text"],
                "reference_code": record["code"],
                "test_setup_code": record.get("test_setup_code", ""),
                "test_list": record.get("test_list", []),
                "challenge_test_list": record.get("challenge_test_list", []),
            }
        )
        if len(code_seeds) >= CODE_TARGET:
            break

    seeds = math_seeds + code_seeds
    plan = {
        "project_goal": "Build a reproducible mini synthetic textbook factory for math and code education.",
        "seed_count": len(seeds),
        "domain_distribution": dict(Counter(seed["domain"] for seed in seeds)),
        "topic_distribution": dict(Counter(seed["topic"] for seed in seeds)),
        "difficulty_distribution": dict(Counter(seed["difficulty"] for seed in seeds)),
        "chapter_templates": build_chapter_templates(),
    }

    write_jsonl(seeds, SEED_FILE)
    write_json(plan, PLAN_FILE)
    print("✅ 种子池与章节模板构建完成。")
    print(plan)


if __name__ == "__main__":
    main()
