#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_LIMIT_MB = 5
DEFAULT_TOTAL_LIMIT_MB = 360
DEFAULT_ROOT = Path("docs/images")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail when documentation images exceed size budgets.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Image directory to scan.")
    parser.add_argument("--limit-mb", type=float, default=DEFAULT_LIMIT_MB, help="Maximum allowed size per image.")
    parser.add_argument(
        "--total-limit-mb",
        type=float,
        default=DEFAULT_TOTAL_LIMIT_MB,
        help="Maximum allowed total size for all scanned images.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("smoke_reports/image_size_report.json"),
        help="Path to write a JSON report.",
    )
    parser.add_argument("--top", type=int, default=20, help="Number of largest images to print.")
    return parser.parse_args()


def scan_images(root: Path) -> list[tuple[Path, int]]:
    return [
        (path, path.stat().st_size)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def main() -> int:
    args = parse_args()
    limit_bytes = int(args.limit_mb * 1024 * 1024)
    total_limit_bytes = int(args.total_limit_mb * 1024 * 1024)
    images = scan_images(args.root)
    total_bytes = sum(size for _, size in images)
    oversized = [(path, size) for path, size in images if size > limit_bytes]
    top_images = sorted(images, key=lambda item: item[1], reverse=True)[: args.top]

    report = {
        "root": str(args.root),
        "image_count": len(images),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1024 / 1024, 3),
        "per_image_limit_mb": args.limit_mb,
        "total_limit_mb": args.total_limit_mb,
        "oversized": [{"path": str(path), "bytes": size} for path, size in oversized],
        "largest_images": [{"path": str(path), "bytes": size} for path, size in top_images],
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Scanned {len(images)} image(s) under {args.root}.")
    print(f"Total image size: {total_bytes / 1024 / 1024:.2f} MB (limit {args.total_limit_mb:g} MB).")
    print(f"Top {len(top_images)} largest image(s):")
    for path, size in top_images:
        print(f"- {path}: {size / 1024 / 1024:.2f} MB")

    failed = False
    if oversized:
        failed = True
        print(f"Found {len(oversized)} image(s) larger than {args.limit_mb:g} MB:")
        for path, size in oversized:
            print(f"- {path}: {size / 1024 / 1024:.2f} MB")
    if total_bytes > total_limit_bytes:
        failed = True
        print(f"Total image size exceeds {args.total_limit_mb:g} MB.")

    if not failed:
        print(f"Image budgets passed: each image <= {args.limit_mb:g} MB and total <= {args.total_limit_mb:g} MB.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
