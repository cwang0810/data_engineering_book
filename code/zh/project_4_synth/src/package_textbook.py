from __future__ import annotations

import os
from collections import Counter, defaultdict

from pipeline_utils import BOOKS_DIR, PROCESSED_DIR, ensure_standard_dirs, load_json, load_jsonl, write_json, write_jsonl

INPUT_FILE = PROCESSED_DIR / "verified_textbook.jsonl"
PLAN_FILE = PROCESSED_DIR / "chapter_plan.json"
CURRICULUM_FILE = PROCESSED_DIR / "curriculum_map.json"
CATALOG_FILE = PROCESSED_DIR / "textbook_catalog.json"
STYLE_GUIDE_FILE = PROCESSED_DIR / "editorial_style_guide.md"
MATH_BOOK_FILE = BOOKS_DIR / "foundations_of_quantitative_reasoning.md"
CODE_BOOK_FILE = BOOKS_DIR / "python_problem_solving_workbook.md"
TEACHER_GUIDE_FILE = BOOKS_DIR / "teacher_guide.md"

MATH_TOPIC_SEQUENCE = [
    "arithmetic_word_problems",
    "fractions_and_proportions",
    "percentages_and_rates",
    "rates_and_motion",
    "geometry_and_measurement",
]
CODE_TOPIC_SEQUENCE = [
    "function_design",
    "lists_and_iteration",
    "string_algorithms",
    "search_and_sort",
    "graphs_and_trees",
]
DIFFICULTY_SEQUENCE = ["core", "advanced", "extension"]

MATH_NOTES = {
    "arithmetic_word_problems": "Start with quantity tracking before introducing more compressed symbolic reasoning.",
    "fractions_and_proportions": "Connect equal partitions to rate tables and unit scaling.",
    "percentages_and_rates": "Reinterpret percentages as multipliers and relative changes.",
    "rates_and_motion": "Use unit rates and repeated intervals to model time-dependent scenarios.",
    "geometry_and_measurement": "Apply formulas only after students identify which dimensions are known.",
}
CODE_NOTES = {
    "function_design": "Teach interface design before optimization so learners can read code contracts clearly.",
    "lists_and_iteration": "Develop traversal habits and accumulator patterns before introducing more compact idioms.",
    "string_algorithms": "Highlight local state updates, boundary checks, and repeated scans.",
    "search_and_sort": "Frame efficiency as a consequence of comparison strategy and loop invariants.",
    "graphs_and_trees": "Introduce recursive structure only after learners are comfortable with helper functions and state.",
}
STOPWORDS = {
    "the", "and", "with", "from", "into", "through", "there", "their", "about", "which", "would",
    "could", "should", "what", "when", "where", "while", "write", "function", "given", "using",
    "find", "calculate", "return", "after", "before", "into", "then", "than", "each", "have",
    "will", "your", "that", "this", "these", "those", "many", "much", "does", "make", "made",
}


def topic_order(domain: str, topic: str) -> int:
    sequence = MATH_TOPIC_SEQUENCE if domain == "math" else CODE_TOPIC_SEQUENCE
    return sequence.index(topic) if topic in sequence else len(sequence)


def difficulty_order(level: str) -> int:
    return DIFFICULTY_SEQUENCE.index(level) if level in DIFFICULTY_SEQUENCE else len(DIFFICULTY_SEQUENCE)


def chapter_learning_objectives(domain: str, topic: str) -> list[str]:
    if domain == "math":
        return [
            "Translate word conditions into explicit arithmetic steps.",
            "Track intermediate quantities without losing the target unknown.",
            "Check the final result against the story context.",
        ]
    return [
        "Read a function prompt and identify the required input-output contract.",
        "Use tests to confirm correctness on normal and edge cases.",
        "Explain why the chosen control flow matches the problem structure.",
    ]


def chapter_key_terms(domain: str, topic: str) -> list[str]:
    if domain == "math":
        if topic == "percentages_and_rates":
            return ["base quantity", "rate", "remaining value", "multiplier"]
        if topic == "rates_and_motion":
            return ["unit rate", "time interval", "total distance", "accumulation"]
        if topic == "fractions_and_proportions":
            return ["ratio", "fraction", "equivalent scaling", "part-whole relation"]
        return ["quantity", "step-by-step reasoning", "intermediate value", "final answer"]
    if topic == "string_algorithms":
        return ["scan", "index", "character test", "state update"]
    if topic == "lists_and_iteration":
        return ["loop", "accumulator", "transformation", "condition"]
    return ["function contract", "test case", "helper variable", "assertion"]


def chapter_common_mistakes(domain: str, topic: str) -> list[str]:
    if domain == "math":
        return [
            "Updating the wrong remaining quantity after an intermediate step.",
            "Mixing percent values with decimal multipliers.",
            "Stopping at an intermediate result instead of the requested final answer.",
        ]
    return [
        "Returning too early before all input elements are processed.",
        "Ignoring an edge case covered by the provided assertions.",
        "Writing code that works on one example but not on the full test set.",
    ]


def prerequisite_topics(domain: str, topic: str) -> list[str]:
    sequence = MATH_TOPIC_SEQUENCE if domain == "math" else CODE_TOPIC_SEQUENCE
    if topic not in sequence:
        return []
    index = sequence.index(topic)
    return sequence[max(0, index - 1):index]


def derive_focus_phrase(text: str) -> str:
    cleaned = "".join(char.lower() if char.isalpha() or char.isspace() else " " for char in text)
    tokens = [token for token in cleaned.split() if len(token) > 3 and token not in STOPWORDS]
    if not tokens:
        return "Core Applications"
    focus = " ".join(tokens[:3])
    return focus.title()


def format_chapter(record: dict) -> str:
    objectives = "\n".join(f"- {item}" for item in record["learning_objectives"])
    prerequisites = ", ".join(item.replace("_", " ").title() for item in record["prerequisites"]) or "None"
    key_terms = ", ".join(record["key_terms"])
    mistakes = "\n".join(f"- {item}" for item in record["common_mistakes"])
    assessment = "\n".join(f"- {item}" for item in record["end_of_chapter_checks"])

    sections = [
        f"## {record['chapter_code']} {record['chapter_title']}",
        "",
        f"Stage: {record['difficulty'].title()}",
        f"Topic: {record['topic'].replace('_', ' ').title()}",
        f"Prerequisites: {prerequisites}",
        "",
        "Learning objectives",
        objectives,
        "",
        "Why this chapter matters",
        record["chapter_rationale"],
        "",
        "Key terms",
        key_terms,
        "",
        "Lesson text",
        record["lesson_text"],
        "",
        "Worked example",
        f"Question: {record.get('worked_example_question', record['exercise_question'])}",
    ]

    if record["domain"] == "math":
        sections.extend(
            [
                f"Solution walkthrough: {record['worked_example_solution']}",
                f"Checkpoint exercise: {record['exercise_question']}",
                f"Verified answer: {record['exercise_answer']}",
            ]
        )
    else:
        sections.extend(
            [
                f"Programming exercise: {record['exercise_question']}",
                "Reference implementation:",
                "```python",
                record["reference_code"],
                "```",
                f"Verification coverage: {len(record['unit_tests'])} assertion(s).",
            ]
        )

    sections.extend(
        [
            "",
            "Common mistakes",
            mistakes,
            "",
            "End-of-chapter checks",
            assessment,
            "",
        ]
    )
    return "\n".join(sections)


def build_book(domain: str, title: str, subtitle: str, chapters: list[dict]) -> str:
    toc_lines = [f"- {record['chapter_code']} {record['chapter_title']}" for record in chapters]
    chapter_blocks = "\n\n".join(format_chapter(record) for record in chapters)
    return (
        f"# {title}\n\n"
        f"{subtitle}\n\n"
        "## Front Matter\n\n"
        "This volume is a reproducible synthetic textbook artifact generated from verified seed tasks. "
        "Each chapter includes objectives, prerequisite tags, a worked example, and end-of-chapter checks.\n\n"
        "## Table Of Contents\n\n"
        + "\n".join(toc_lines)
        + "\n\n## Main Text\n\n"
        + chapter_blocks
        + "\n"
    )


def main() -> None:
    ensure_standard_dirs()
    plan = load_json(PLAN_FILE)
    records = load_jsonl(INPUT_FILE)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["domain"]].append(record)

    curriculum = {
        "volumes": [],
        "chapter_count": len(records),
        "domain_distribution": dict(Counter(record["domain"] for record in records)),
    }

    enriched_records: list[dict] = []
    catalog = {"books": []}
    book_specs = [
        ("math", "Volume I. Foundations of Quantitative Reasoning", "A compact workbook for arithmetic, rates, and proportional reasoning.", MATH_BOOK_FILE),
        ("code", "Volume II. Python Problem-Solving Workbook", "A structured beginner-to-intermediate programming text with verified examples.", CODE_BOOK_FILE),
    ]

    for volume_index, (domain, title, subtitle, output_path) in enumerate(book_specs, start=1):
        chapter_records = sorted(
            grouped.get(domain, []),
            key=lambda item: (topic_order(domain, item["topic"]), difficulty_order(item["difficulty"]), item["id"]),
        )
        for chapter_index, record in enumerate(chapter_records, start=1):
            topic_note = MATH_NOTES.get(record["topic"], "") if domain == "math" else CODE_NOTES.get(record["topic"], "")
            base_title = record["topic"].replace("_", " ").title()
            focus_phrase = derive_focus_phrase(record.get("exercise_question", record.get("worked_example_question", "")))
            record["volume"] = volume_index
            record["chapter_number"] = chapter_index
            record["chapter_code"] = f"Chapter {chapter_index:02d}"
            record["chapter_title"] = f"{base_title}: {focus_phrase}"
            record["prerequisites"] = prerequisite_topics(domain, record["topic"])
            record["learning_objectives"] = chapter_learning_objectives(domain, record["topic"])
            record["key_terms"] = chapter_key_terms(domain, record["topic"])
            record["common_mistakes"] = chapter_common_mistakes(domain, record["topic"])
            record["chapter_rationale"] = topic_note
            record["end_of_chapter_checks"] = [
                "State the main idea of the chapter in one sentence.",
                "Re-solve the checkpoint exercise without looking at the example.",
                "Explain one mistake that would lead to a wrong answer.",
            ]
            enriched_records.append(record)

        book_text = build_book(domain, title, subtitle, chapter_records)
        if os.environ.get("PROJECT_SMOKE") != "1" or not output_path.exists():
            output_path.write_text(book_text, encoding="utf-8")
        catalog["books"].append(
            {
                "domain": domain,
                "title": title,
                "subtitle": subtitle,
                "path": output_path.relative_to(output_path.parent.parent).as_posix(),
                "num_chapters": len(chapter_records),
            }
        )
        curriculum["volumes"].append(
            {
                "volume": volume_index,
                "domain": domain,
                "title": title,
                "subtitle": subtitle,
                "num_chapters": len(chapter_records),
                "topics_in_order": [record["topic"] for record in chapter_records],
            }
        )

    style_guide = """# Editorial Style Guide

## Positioning

- Write as if this were a concise printed workbook for guided self-study.
- Keep tone instructional, calm, and explicit.
- Prefer short paragraphs, numbered steps, and verified outcomes.

## Chapter Structure

- Every chapter should declare prerequisites and learning objectives.
- Every worked example must end with a checked answer.
- Every code chapter should mention verification through assertions.
- Every chapter should include common mistakes and end-of-chapter checks.

## Difficulty Policy

- `core`: direct application with one main idea.
- `advanced`: multi-step reasoning with a hidden intermediate quantity.
- `extension`: asks learners to connect ideas across steps or abstractions.
"""
    STYLE_GUIDE_FILE.write_text(style_guide, encoding="utf-8")

    teacher_guide = (
        "# Teacher Guide\n\n"
        f"Seed count: {plan['seed_count']}\n\n"
        "Recommended usage:\n"
        "- Use the math volume for model lessons on quantitative reasoning.\n"
        "- Use the code volume for lab sessions centered on assertions and debugging.\n"
        "- Assign one checkpoint exercise and one end-of-chapter reflection for each lesson.\n"
    )
    if os.environ.get("PROJECT_SMOKE") != "1" or not TEACHER_GUIDE_FILE.exists():
        TEACHER_GUIDE_FILE.write_text(teacher_guide, encoding="utf-8")

    write_json(curriculum, CURRICULUM_FILE)
    write_json(catalog, CATALOG_FILE)
    write_jsonl(enriched_records, INPUT_FILE)
    print("✅ 教材目录、先修关系和成册输出生成完成。")
    print(
        {
            "num_books": len(catalog["books"]),
            "num_chapters": len(enriched_records),
        }
    )


if __name__ == "__main__":
    main()
