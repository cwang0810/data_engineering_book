#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


DEFAULT_ROOT = Path("docs/zh")
STYLE_TERMS = {
    "毒打": "强化训练、压力测试",
    "暴打": "压力测试、对比验证",
    "屁股": "输入端、后续处理链路",
    "凶萌": "典型、具有代表性",
    "血肉": "高噪声、复杂",
    "惨烈": "代价高、风险高",
    "变态": "复杂、高强度",
    "毛骨悚然": "异常、严重",
    "大魔王": "高难度样本、关键挑战",
    "背锅": "承担根因、成为问题来源",
    "极其恐怖": "显著、严重",
    "核聚变": "显著提升",
    "大杀四方": "表现显著提升",
    "终局之战": "收束章节、关键阶段",
    "疯狂万能巨兽": "缺少约束的通用系统",
    "尸骨累累": "失败案例较多",
    "黑暗堡垒": "尚未解决的挑战",
    "傻子": "能力不足的系统",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check high-risk wording in Chinese chapters.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Directory to scan.")
    parser.add_argument("--baseline", type=Path, help="JSON baseline. Fails when counts exceed it.")
    parser.add_argument("--write-baseline", type=Path, help="Write current counts as a baseline and exit.")
    return parser.parse_args()


def scan(root: Path) -> tuple[Counter[str], list[dict]]:
    counts: Counter[str] = Counter()
    hits: list[dict] = []
    for path in sorted(root.rglob("*.md")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for term, suggestion in STYLE_TERMS.items():
                count = line.count(term)
                if count:
                    counts[term] += count
                    hits.append(
                        {
                            "path": path.as_posix(),
                            "line": lineno,
                            "term": term,
                            "count": count,
                            "suggestion": suggestion,
                            "text": line.strip(),
                        }
                    )
    return counts, hits


def load_baseline(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "term_counts" in payload:
        return {str(key): int(value) for key, value in payload["term_counts"].items()}
    return {str(key): int(value) for key, value in payload.items()}


def main() -> int:
    args = parse_args()
    counts, hits = scan(args.root)

    if args.write_baseline:
        payload = {
            "root": args.root.as_posix(),
            "term_counts": dict(sorted(counts.items())),
            "total_hits": sum(counts.values()),
        }
        args.write_baseline.parent.mkdir(parents=True, exist_ok=True)
        args.write_baseline.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote style baseline to {args.write_baseline}.")
        return 0

    print(f"Scanned {args.root}. High-risk term hits: {sum(counts.values())}.")
    for hit in hits:
        print(
            f"{hit['path']}:{hit['line']}: {hit['term']} x{hit['count']} "
            f"(suggest: {hit['suggestion']})"
        )

    if not args.baseline:
        return 0 if not hits else 1

    baseline = load_baseline(args.baseline)
    regressions = {
        term: count
        for term, count in counts.items()
        if count > baseline.get(term, 0)
    }
    missing_terms = {
        term: 0
        for term in baseline
        if term not in STYLE_TERMS
    }
    if missing_terms:
        print(f"Baseline contains unknown term(s): {', '.join(sorted(missing_terms))}", file=sys.stderr)
        return 1

    if regressions:
        print("Style term regression detected:", file=sys.stderr)
        for term, count in sorted(regressions.items()):
            print(f"- {term}: {count} > baseline {baseline.get(term, 0)}", file=sys.stderr)
        return 1

    print("Style term counts are within baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
