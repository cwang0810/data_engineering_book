from __future__ import annotations

from collections import Counter

from pipeline_utils import PROCESSED_DIR, corrupt_numeric_text, ensure_standard_dirs, load_jsonl, mutate_python_code, write_json, write_jsonl

SEED_FILE = PROCESSED_DIR / "seed_pool.jsonl"
TRACE_FILE = PROCESSED_DIR / "cot_traces.jsonl"
SUMMARY_FILE = PROCESSED_DIR / "trace_summary.json"


def build_math_traces(seed: dict) -> list[dict]:
    correct_steps = [
        {"step_idx": idx + 1, "text": step, "label": 1, "kind": "reasoning"}
        for idx, step in enumerate(seed["reference_steps"])
    ]
    final_step = {
        "step_idx": len(correct_steps) + 1,
        "text": f"Final answer: {seed['final_answer']}",
        "label": 1,
        "kind": "final",
    }
    positive = {
        "trace_id": f"{seed['seed_id']}_positive",
        "seed_id": seed["seed_id"],
        "domain": seed["domain"],
        "topic": seed["topic"],
        "trace_type": "positive",
        "question": seed["question"],
        "steps": correct_steps + [final_step],
        "candidate_answer": seed["final_answer"],
        "expected_answer": seed["final_answer"],
        "source_dataset": seed["source_dataset"],
    }

    wrong_steps = [dict(step) for step in correct_steps]
    wrong_steps[-1]["text"] = corrupt_numeric_text(wrong_steps[-1]["text"])
    wrong_steps[-1]["label"] = 0
    negative_final = {
        "step_idx": len(wrong_steps) + 1,
        "text": f"Final answer: {corrupt_numeric_text(seed['final_answer'])}",
        "label": 0,
        "kind": "final",
    }
    negative = {
        "trace_id": f"{seed['seed_id']}_negative",
        "seed_id": seed["seed_id"],
        "domain": seed["domain"],
        "topic": seed["topic"],
        "trace_type": "negative",
        "question": seed["question"],
        "steps": wrong_steps + [negative_final],
        "candidate_answer": negative_final["text"].split(":", 1)[1].strip(),
        "expected_answer": seed["final_answer"],
        "source_dataset": seed["source_dataset"],
    }

    repair_steps = wrong_steps + [negative_final]
    repair_steps.append(
        {
            "step_idx": len(repair_steps) + 1,
            "text": f"Correction: the previous arithmetic was wrong. The correct final answer is {seed['final_answer']}.",
            "label": 1,
            "kind": "repair",
        }
    )
    repair = {
        "trace_id": f"{seed['seed_id']}_repair",
        "seed_id": seed["seed_id"],
        "domain": seed["domain"],
        "topic": seed["topic"],
        "trace_type": "repair",
        "question": seed["question"],
        "steps": repair_steps,
        "candidate_answer": seed["final_answer"],
        "expected_answer": seed["final_answer"],
        "source_dataset": seed["source_dataset"],
    }
    return [positive, negative, repair]


def build_code_traces(seed: dict) -> list[dict]:
    tests = seed["test_list"] + seed.get("challenge_test_list", [])
    correct_steps = [
        {
            "step_idx": 1,
            "text": f"Understand the task: {seed['question']}",
            "label": 1,
            "kind": "analysis",
        },
        {
            "step_idx": 2,
            "text": f"Select a solution strategy for topic `{seed['topic']}` and prepare to validate with assertions.",
            "label": 1,
            "kind": "planning",
        },
        {
            "step_idx": 3,
            "text": seed["reference_code"],
            "label": 1,
            "kind": "code",
        },
        {
            "step_idx": 4,
            "text": f"Run {len(tests)} unit tests to confirm the implementation.",
            "label": 1,
            "kind": "verification",
        },
    ]
    positive = {
        "trace_id": f"{seed['seed_id']}_positive",
        "seed_id": seed["seed_id"],
        "domain": seed["domain"],
        "topic": seed["topic"],
        "trace_type": "positive",
        "question": seed["question"],
        "steps": correct_steps,
        "candidate_code": seed["reference_code"],
        "reference_code": seed["reference_code"],
        "test_setup_code": seed["test_setup_code"],
        "unit_tests": tests,
        "source_dataset": seed["source_dataset"],
    }

    broken_code = mutate_python_code(seed["reference_code"])
    negative_steps = [
        dict(correct_steps[0]),
        dict(correct_steps[1]),
        {
            "step_idx": 3,
            "text": broken_code,
            "label": 0,
            "kind": "code",
        },
        {
            "step_idx": 4,
            "text": "The implementation compiles conceptually, but the tests reveal a logic error.",
            "label": 0,
            "kind": "verification",
        },
    ]
    negative = {
        "trace_id": f"{seed['seed_id']}_negative",
        "seed_id": seed["seed_id"],
        "domain": seed["domain"],
        "topic": seed["topic"],
        "trace_type": "negative",
        "question": seed["question"],
        "steps": negative_steps,
        "candidate_code": broken_code,
        "reference_code": seed["reference_code"],
        "test_setup_code": seed["test_setup_code"],
        "unit_tests": tests,
        "source_dataset": seed["source_dataset"],
    }

    repair_steps = negative_steps + [
        {
            "step_idx": 5,
            "text": "Correction: revert the broken logic and restore the reference implementation before rerunning the tests.",
            "label": 1,
            "kind": "repair",
        }
    ]
    repair = {
        "trace_id": f"{seed['seed_id']}_repair",
        "seed_id": seed["seed_id"],
        "domain": seed["domain"],
        "topic": seed["topic"],
        "trace_type": "repair",
        "question": seed["question"],
        "steps": repair_steps,
        "candidate_code": seed["reference_code"],
        "reference_code": seed["reference_code"],
        "test_setup_code": seed["test_setup_code"],
        "unit_tests": tests,
        "source_dataset": seed["source_dataset"],
    }
    return [positive, negative, repair]


def main() -> None:
    ensure_standard_dirs()
    seeds = load_jsonl(SEED_FILE)
    traces: list[dict] = []

    for seed in seeds:
        if seed["domain"] == "math":
            traces.extend(build_math_traces(seed))
        else:
            traces.extend(build_code_traces(seed))

    summary = {
        "num_traces": len(traces),
        "trace_type_distribution": dict(Counter(trace["trace_type"] for trace in traces)),
        "domain_distribution": dict(Counter(trace["domain"] for trace in traces)),
        "step_label_distribution": dict(Counter(step["label"] for trace in traces for step in trace["steps"])),
    }

    write_jsonl(traces, TRACE_FILE)
    write_json(summary, SUMMARY_FILE)
    print("✅ P6 推理轨迹生成完成。")
    print(summary)


if __name__ == "__main__":
    main()
