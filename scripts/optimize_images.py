#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Optional

from PIL import Image


DEFAULT_ROOT = Path("docs/images")
DEFAULT_MIN_SIZE_MB = 2.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize large PNG assets in-place while preserving filenames.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Image directory to scan.")
    parser.add_argument("--min-size-mb", type=float, default=DEFAULT_MIN_SIZE_MB, help="Only optimize PNGs above this size.")
    parser.add_argument("--colors", type=int, default=256, help="Palette size for PNG quantization.")
    parser.add_argument("--lossless-only", action="store_true", help="Skip palette quantization and only repack PNG files.")
    parser.add_argument("--dry-run", action="store_true", help="Print candidates without modifying files.")
    return parser.parse_args()


def _save_lossless_candidate(image: Image.Image, target: Path) -> None:
    image.save(target, format="PNG", optimize=True, compress_level=9)


def _save_quantized_candidate(image: Image.Image, target: Path, colors: int) -> None:
    if image.mode == "RGBA":
        quantized = image.quantize(colors=colors, method=Image.Quantize.FASTOCTREE)
    elif image.mode == "RGB":
        quantized = image.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    else:
        quantized = image.convert("RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    quantized.save(target, format="PNG", optimize=True, compress_level=9)


def optimize_png(path: Path, dry_run: bool, colors: int, lossless_only: bool) -> tuple[int, int, str]:
    before = path.stat().st_size
    if dry_run:
        return before, before, "dry-run"

    best_path: Optional[Path] = None
    best_size = before
    best_method = "kept"

    with Image.open(path) as image:
        with tempfile.TemporaryDirectory() as tmpdir:
            lossless_path = Path(tmpdir) / "lossless.png"
            _save_lossless_candidate(image, lossless_path)
            lossless_size = lossless_path.stat().st_size
            if lossless_size < best_size:
                best_path = lossless_path
                best_size = lossless_size
                best_method = "lossless"

            if not lossless_only:
                quantized_path = Path(tmpdir) / "quantized.png"
                _save_quantized_candidate(image, quantized_path, colors)
                quantized_size = quantized_path.stat().st_size
                if quantized_size < best_size:
                    best_path = quantized_path
                    best_size = quantized_size
                    best_method = f"quantized-{colors}"

            if best_path is not None:
                path.write_bytes(best_path.read_bytes())

    return before, path.stat().st_size, best_method


def main() -> int:
    args = parse_args()
    min_bytes = int(args.min_size_mb * 1024 * 1024)
    candidates = [
        path
        for path in sorted(args.root.rglob("*.png"))
        if path.is_file() and path.stat().st_size > min_bytes
    ]

    total_before = 0
    total_after = 0
    for path in candidates:
        before, after, method = optimize_png(path, args.dry_run, args.colors, args.lossless_only)
        total_before += before
        total_after += after
        delta = before - after
        print(
            f"{path}: {before / 1024 / 1024:.2f} MB -> {after / 1024 / 1024:.2f} MB "
            f"({delta / 1024 / 1024:.2f} MB saved, {method})"
        )

    print(f"Optimized {len(candidates)} PNG candidate(s).")
    if candidates:
        print(f"Total: {total_before / 1024 / 1024:.2f} MB -> {total_after / 1024 / 1024:.2f} MB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
