from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TRAINING_DIR = DATA_DIR / "training"
REPORTS_DIR = DATA_DIR / "reports"

P4_DATA_DIR = ROOT_DIR.parent / "project_4_synth" / "data"
GSM8K_FILE = P4_DATA_DIR / "gsm8k_train.jsonl"
MBPP_FILE = P4_DATA_DIR / "mbpp_train.jsonl"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_standard_dirs() -> None:
    for path in [PROCESSED_DIR, TRAINING_DIR, REPORTS_DIR]:
        ensure_dir(path)


def write_json(data, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(records: list[dict], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def deterministic_bucket(key: str, buckets: int = 100) -> int:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


def estimated_tokens(text: str) -> int:
    compact = normalize_text(text)
    return max(1, len(compact) // 4)


def clean_gsm8k_answer(answer: str) -> str:
    answer = re.sub(r"<<([^>]+)>>", r"\1", answer)
    return answer.strip()


def extract_final_answer(answer: str) -> str | None:
    match = re.search(r"####\s*(.+)$", answer, re.MULTILINE)
    if match:
        return normalize_text(match.group(1))
    return None


def split_reasoning_steps(answer: str) -> list[str]:
    cleaned = clean_gsm8k_answer(answer)
    lines = [normalize_text(line) for line in cleaned.splitlines() if normalize_text(line)]
    steps = []
    for line in lines:
        if line.startswith("####"):
            continue
        steps.append(line)
    return steps


def first_number(text: str) -> str | None:
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", text)
    return match.group(0) if match else None


def corrupt_numeric_text(text: str) -> str:
    number = first_number(text)
    if not number:
        return text + " Therefore the final answer is 0."
    raw = number.replace(",", "")
    if "." in raw:
        new_value = str(round(float(raw) + 1.0, 2))
    else:
        new_value = str(int(raw) + 1)
    return text.replace(number, new_value, 1)


def mutate_python_code(code: str) -> str:
    for pattern, replacement in [
        (r"return\s+([^\n]+)", "return None"),
        (r"==", "!="),
        (r"\+", "-"),
    ]:
        mutated = re.sub(pattern, replacement, code, count=1)
        if mutated != code:
            return mutated
    return code + "\nraise AssertionError('broken trace')\n"


def run_python_snippet(code: str, timeout: int = 5) -> tuple[bool, str, str]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        script = Path(tmp_dir) / "script.py"
        script.write_text(code, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    return completed.returncode == 0, completed.stdout.strip(), completed.stderr.strip()


def reward_bucket(score: float) -> str:
    if score >= 0.95:
        return "high"
    if score >= 0.6:
        return "medium"
    if score > 0:
        return "low"
    return "fail"
