from __future__ import annotations

from collections import Counter

from pipeline_utils import PROCESSED_DIR, ensure_standard_dirs, load_jsonl, reward_bucket, run_python_snippet, write_json, write_jsonl

TRACE_FILE = PROCESSED_DIR / "cot_traces.jsonl"
VALIDATED_FILE = PROCESSED_DIR / "validated_traces.jsonl"
STEP_REWARD_FILE = PROCESSED_DIR / "step_rewards.jsonl"
SUMMARY_FILE = PROCESSED_DIR / "validation_summary.json"


def validate_math_trace(trace: dict) -> tuple[bool, dict]:
    expected = str(trace["expected_answer"]).strip()
    candidate = str(trace["candidate_answer"]).strip()
    passed = expected == candidate
    return passed, {
        "validation_mode": "final_answer_match",
        "expected_answer": expected,
        "candidate_answer": candidate,
        "stdout": candidate,
        "stderr": "",
    }


def validate_code_trace(trace: dict) -> tuple[bool, dict]:
    code_parts = [
        trace.get("test_setup_code", ""),
        trace["candidate_code"],
        "\n".join(trace["unit_tests"]),
    ]
    code = "\n\n".join(part for part in code_parts if part).strip() + "\n"
    passed, stdout, stderr = run_python_snippet(code)
    return passed, {
        "validation_mode": "execution_and_unit_tests",
        "expected_answer": "all_tests_pass",
        "candidate_answer": "all_tests_pass" if passed else "test_failure",
        "stdout": stdout,
        "stderr": stderr,
    }


def main() -> None:
    ensure_standard_dirs()
    traces = load_jsonl(TRACE_FILE)
    validated: list[dict] = []
    step_rewards: list[dict] = []

    for trace in traces:
        if trace["domain"] == "math":
            passed, validation = validate_math_trace(trace)
        else:
            passed, validation = validate_code_trace(trace)

        label_sum = sum(step["label"] for step in trace["steps"])
        score = label_sum / max(1, len(trace["steps"]))
        bucket = reward_bucket(score)

        enriched = dict(trace)
        enriched["validation"] = validation
        enriched["validation_passed"] = passed
        enriched["trace_score"] = round(score, 4)
        enriched["reward_bucket"] = bucket
        validated.append(enriched)

        for step in trace["steps"]:
            step_rewards.append(
                {
                    "trace_id": trace["trace_id"],
                    "seed_id": trace["seed_id"],
                    "domain": trace["domain"],
                    "trace_type": trace["trace_type"],
                    "step_idx": step["step_idx"],
                    "step_text": step["text"],
                    "step_kind": step["kind"],
                    "label": step["label"],
                    "trace_score": round(score, 4),
                    "reward_bucket": bucket,
                    "validation_passed": passed,
                }
            )

    summary = {
        "num_traces": len(validated),
        "validation_pass_rate": round(sum(item["validation_passed"] for item in validated) / max(1, len(validated)), 4),
        "trace_type_distribution": dict(Counter(item["trace_type"] for item in validated)),
        "domain_distribution": dict(Counter(item["domain"] for item in validated)),
        "reward_bucket_distribution": dict(Counter(item["reward_bucket"] for item in validated)),
    }

    write_jsonl(validated, VALIDATED_FILE)
    write_jsonl(step_rewards, STEP_REWARD_FILE)
    write_json(summary, SUMMARY_FILE)
    print("✅ P6 自动验证与打分完成。")
    print(summary)


if __name__ == "__main__":
    main()
