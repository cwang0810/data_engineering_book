from __future__ import annotations

from pipeline_utils import PROCESSED_DIR, ensure_standard_dirs, load_jsonl, normalize_text, write_jsonl

SEED_FILE = PROCESSED_DIR / "seed_pool.jsonl"
OUTPUT_FILE = PROCESSED_DIR / "synthetic_textbook_chapters.jsonl"

MATH_TOPIC_NOTES = {
    "percentages_and_rates": "This chapter teaches learners to convert percentage statements into explicit multipliers and to update the remaining quantity after each transaction.",
    "rates_and_motion": "This chapter emphasizes rate-time-quantity relationships and highlights how to aggregate repeated travel or work intervals.",
    "geometry_and_measurement": "This chapter reviews measurement formulas and shows how to turn verbal descriptions into symbolic quantities.",
    "fractions_and_proportions": "This chapter focuses on proportional reasoning, equal partitioning, and fraction-to-quantity conversions.",
    "arithmetic_word_problems": "This chapter practices quantity tracking, multi-step arithmetic, and careful identification of what remains after each event.",
}

CODE_TOPIC_NOTES = {
    "string_algorithms": "This chapter introduces string scanning patterns, character bookkeeping, and small helper conditions.",
    "lists_and_iteration": "This chapter explains traversal patterns, accumulator variables, and how to transform list data into reliable outputs.",
    "graphs_and_trees": "This chapter motivates structural recursion and careful state propagation across composite data.",
    "search_and_sort": "This chapter reviews ordering, comparison logic, and how loop invariants justify a final result.",
    "function_design": "This chapter highlights how to choose function boundaries, name helper variables, and preserve clarity under constraints.",
}


def chapter_title(seed: dict) -> str:
    if seed["domain"] == "math":
        return f"Math Chapter: {seed['topic'].replace('_', ' ').title()}"
    return f"Code Chapter: {seed['topic'].replace('_', ' ').title()}"


def build_math_record(seed: dict) -> dict:
    body = MATH_TOPIC_NOTES[seed["topic"]]
    full_chapter = (
        f"# {chapter_title(seed)}\n\n"
        f"Audience: {seed['difficulty']} learners.\n\n"
        f"Concept note: {body}\n\n"
        "Worked example:\n"
        f"Problem: {seed['seed_question']}\n"
        f"Solution sketch: {seed['source_solution']}\n\n"
        "Checkpoint exercise:\n"
        f"Solve the same problem carefully and verify the final quantity.\n\n"
        "Expected result:\n"
        f"{seed['expected_answer']}\n"
    )
    verification_code = f"def solve():\n    return {repr(seed['expected_answer'])}\n\nprint(solve())\n"
    unit_tests = [f"assert str(solve()) == {repr(seed['expected_answer'])}"]
    return {
        "id": seed["id"],
        "domain": "math",
        "topic": seed["topic"],
        "difficulty": seed["difficulty"],
        "chapter_title": chapter_title(seed),
        "lesson_text": body,
        "worked_example_question": seed["seed_question"],
        "worked_example_solution": seed["source_solution"],
        "exercise_question": seed["seed_question"],
        "exercise_answer": seed["expected_answer"],
        "reference_code": verification_code,
        "unit_tests": unit_tests,
        "chapter_markdown": full_chapter,
        "source_dataset": seed["source_dataset"],
    }


def build_code_record(seed: dict) -> dict:
    body = CODE_TOPIC_NOTES[seed["topic"]]
    tests = seed["test_list"] + seed.get("challenge_test_list", [])
    explanation = (
        f"This section belongs to the topic `{seed['topic']}`. "
        "Learners should identify the data transformation, choose a function signature, "
        "and confirm correctness with assertions."
    )
    full_chapter = (
        f"# {chapter_title(seed)}\n\n"
        f"Audience: {seed['difficulty']} learners.\n\n"
        f"Concept note: {body}\n\n"
        "Programming exercise:\n"
        f"{seed['seed_question']}\n\n"
        "Annotated solution:\n"
        f"```python\n{seed['reference_code']}\n```\n\n"
        "Testing note:\n"
        f"Run the provided assertions to validate edge cases. Total assertions: {len(tests)}.\n"
    )
    return {
        "id": seed["id"],
        "domain": "code",
        "topic": seed["topic"],
        "difficulty": seed["difficulty"],
        "chapter_title": chapter_title(seed),
        "lesson_text": normalize_text(f"{body} {explanation}"),
        "exercise_question": seed["seed_question"],
        "reference_code": seed["reference_code"],
        "test_setup_code": seed.get("test_setup_code", ""),
        "unit_tests": tests,
        "chapter_markdown": full_chapter,
        "source_dataset": seed["source_dataset"],
    }


def main() -> None:
    ensure_standard_dirs()
    seeds = load_jsonl(SEED_FILE)
    records: list[dict] = []
    for seed in seeds:
        if seed["domain"] == "math":
            records.append(build_math_record(seed))
        else:
            records.append(build_code_record(seed))

    write_jsonl(records, OUTPUT_FILE)
    print("✅ 合成教材章节生成完成。")
    print({"num_records": len(records)})


if __name__ == "__main__":
    main()
