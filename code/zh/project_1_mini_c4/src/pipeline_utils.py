from __future__ import annotations

import hashlib
import re
from statistics import median
from urllib.parse import urlparse

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"\d")
WORD_RE = re.compile(r"[A-Za-z0-9']+")
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", " ")

    cleaned_lines = []
    previous_line = None
    for raw_line in text.split("\n"):
        line = WHITESPACE_RE.sub(" ", raw_line).strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            previous_line = ""
            continue

        if line == previous_line:
            continue

        cleaned_lines.append(line)
        previous_line = line

    normalized = "\n".join(cleaned_lines).strip()
    return MULTI_NEWLINE_RE.sub("\n\n", normalized)


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def extract_domain(url: str) -> str:
    if not url:
        return "unknown"

    parsed = urlparse(url)
    host = parsed.netloc.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host or "unknown"


def estimate_token_count(text: str) -> int:
    cjk_count = len(CJK_RE.findall(text))
    latin_words = len(WORD_RE.findall(text))
    return cjk_count + latin_words


def script_counts(text: str) -> dict[str, int]:
    return {
        "cjk": len(CJK_RE.findall(text)),
        "latin": len(LATIN_RE.findall(text)),
        "digits": len(DIGIT_RE.findall(text)),
    }


def split_nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def duplicate_line_ratio(lines: list[str]) -> float:
    if not lines:
        return 0.0
    unique_count = len(set(lines))
    return 1.0 - unique_count / len(lines)


def short_line_ratio(lines: list[str], max_len: int = 32) -> float:
    if not lines:
        return 0.0
    return sum(1 for line in lines if len(line) <= max_len) / len(lines)


def median_line_length(lines: list[str]) -> float:
    if not lines:
        return 0.0
    return float(median(len(line) for line in lines))


def character_ratio(text: str, charset: str) -> float:
    if not text:
        return 0.0
    if charset == "digits":
        count = len(DIGIT_RE.findall(text))
    elif charset == "latin":
        count = len(LATIN_RE.findall(text))
    elif charset == "cjk":
        count = len(CJK_RE.findall(text))
    else:
        raise ValueError(f"Unsupported charset: {charset}")
    return count / len(text)
