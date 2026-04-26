from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from pipeline_utils import PROCESSED_DIR, ensure_standard_dirs, load_jsonl, write_jsonl

INPUT_FILE = PROCESSED_DIR / "synthetic_textbook_chapters.jsonl"
VERIFIED_FILE = PROCESSED_DIR / "verified_textbook.jsonl"
FAILURE_FILE = PROCESSED_DIR / "verification_failures.jsonl"
RESULTS_FILE = PROCESSED_DIR / "execution_results.jsonl"


def run_python(code: str, timeout: int = 5) -> tuple[bool, str, str]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = Path(tmp_dir) / "script.py"
        script_path.write_text(code, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    return result.returncode == 0, result.stdout.strip(), result.stderr.strip()


def verify_math(record: dict) -> tuple[bool, dict]:
    code = record["reference_code"] + "\n" + "\n".join(record["unit_tests"]) + "\n"
    passed, stdout, stderr = run_python(code)
    metadata = {
        "id": record["id"],
        "domain": record["domain"],
        "passed": passed,
        "stdout": stdout,
        "stderr": stderr,
        "num_tests": len(record["unit_tests"]),
    }
    return passed, metadata


def verify_code(record: dict) -> tuple[bool, dict]:
    code_parts = [
        record.get("test_setup_code", ""),
        record["reference_code"],
        "\n".join(record["unit_tests"]),
    ]
    code = "\n\n".join(part for part in code_parts if part).strip() + "\n"
    passed, stdout, stderr = run_python(code)
    metadata = {
        "id": record["id"],
        "domain": record["domain"],
        "passed": passed,
        "stdout": stdout,
        "stderr": stderr,
        "num_tests": len(record["unit_tests"]),
    }
    return passed, metadata


def main() -> None:
    ensure_standard_dirs()
    records = load_jsonl(INPUT_FILE)
    verified: list[dict] = []
    failures: list[dict] = []
    execution_results: list[dict] = []

    for record in records:
        passed, metadata = verify_math(record) if record["domain"] == "math" else verify_code(record)
        execution_results.append(metadata)
        enriched = dict(record)
        enriched["verification"] = metadata
        if passed:
            verified.append(enriched)
        else:
            failures.append(enriched)

    write_jsonl(verified, VERIFIED_FILE)
    write_jsonl(failures, FAILURE_FILE)
    write_jsonl(execution_results, RESULTS_FILE)
    print("✅ 程序执行与单元测试验证完成。")
    print(
        {
            "num_records": len(records),
            "verified_records": len(verified),
            "failed_records": len(failures),
        }
    )


if __name__ == "__main__":
    main()
