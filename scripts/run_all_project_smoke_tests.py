#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATED_DIRS = (
    "data/processed",
    "data/training",
    "data/reports",
    "data/eval",
    "data/console",
    "data/page_images",
    "data/derived",
    "data/qa_viz",
)


@dataclass(frozen=True)
class Project:
    project_id: str
    root: Path
    check_script: str
    prepare_scripts: tuple[str, ...]
    generated_dirs: tuple[str, ...] = DEFAULT_GENERATED_DIRS


PROJECTS = [
    Project(
        "P1",
        ROOT / "code/zh/project_1_mini_c4",
        "src/run_p1_checks.py",
        (
            "src/1_download_data.py",
            "src/2_process_warc.py",
            "src/3_clean_data.py",
            "src/4_deduplicate.py",
            "src/5_split_lang.py",
            "src/6_quality_filter.py",
            "src/7_prepare_training_data.py",
            "src/8_evaluate_dataset.py",
            "src/9_training_smoke_test.py",
        ),
    ),
    Project(
        "P2",
        ROOT / "code/zh/project_2_sft_data",
        "src/run_p2_checks.py",
        (
            "src/data_processing.py",
            "src/generate_instructions.py",
            "src/build_preference_data.py",
            "src/prepare_training_data.py",
            "src/downstream_validation.py",
            "src/evaluate_factory.py",
        ),
    ),
    Project(
        "P3",
        ROOT / "code/zh/project_3_llava_data",
        "src/run_p3_checks.py",
        (
            "src/collect_multimodal_assets.py",
            "src/alignment.py",
            "src/generate_llava_data.py",
            "src/interleaved.py",
            "src/quality_control.py",
            "src/visualize_bbox.py",
            "src/prepare_training_data.py",
            "src/evaluate_factory.py",
        ),
    ),
    Project(
        "P4",
        ROOT / "code/zh/project_4_synth",
        "src/run_p4_checks.py",
        (
            "src/sampler.py",
            "src/evol.py",
            "src/sandbox.py",
            "src/package_textbook.py",
            "src/quality_control.py",
            "src/prepare_training_data.py",
            "src/evaluate_factory.py",
        ),
    ),
    Project(
        "P5",
        ROOT / "code/zh/project_5_rag",
        "src/run_p5_checks.py",
        (
            "src/index.py",
            "src/evaluate_rag.py",
        ),
    ),
    Project(
        "P6",
        ROOT / "code/zh/project_6_prm_data",
        "src/run_p6_checks.py",
        (
            "src/sampler.py",
            "src/generate_traces.py",
            "src/validate_and_score.py",
            "src/prepare_prm_data.py",
            "src/evaluate_prm.py",
        ),
    ),
    Project(
        "P7",
        ROOT / "code/zh/project_7_agent_tooluse",
        "src/run_p7_checks.py",
        (
            "src/build_tooling.py",
            "src/generate_trajectories.py",
            "src/simulate_tool_env.py",
            "src/prepare_agent_dataset.py",
            "src/evaluate_tooluse.py",
        ),
    ),
    Project(
        "P8",
        ROOT / "code/zh/project_8_dataops_platform",
        "src/run_p8_checks.py",
        (
            "src/build_platform_specs.py",
            "src/simulate_platform_ops.py",
            "src/evaluate_platform.py",
        ),
    ),
    Project(
        "P9",
        ROOT / "code/zh/project_9_privacy_pipeline",
        "src/run_p9_checks.py",
        (
            "src/build_privacy_specs.py",
            "src/run_privacy_pipeline.py",
            "src/simulate_privacy_ops.py",
            "src/evaluate_privacy_pipeline.py",
        ),
    ),
    Project(
        "P10",
        ROOT / "code/zh/project_10_llm_flywheel",
        "src/run_p10_checks.py",
        (
            "src/collect_upstream_projects.py",
            "src/build_flywheel.py",
            "src/evaluate_flywheel.py",
        ),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all project smoke checks and write a summary report.")
    parser.add_argument("--project", action="append", help="Project id to run, such as P1. Can be repeated.")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per command in seconds.")
    parser.add_argument("--full", action="store_true", help="Run each project's local fixture pipeline before checks.")
    parser.add_argument("--clean", action="store_true", help="Remove generated project artifacts before full checks.")
    parser.add_argument("--no-prepare", action="store_true", help="In full mode, skip preparation and run checks only.")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "smoke_reports")
    return parser.parse_args()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def clean_generated_dirs(project: Project) -> list[str]:
    removed: list[str] = []
    for dirname in project.generated_dirs:
        target = project.root / dirname
        if target.exists():
            shutil.rmtree(target)
            removed.append(rel(target))
    return removed


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_p2_fixture(project: Project) -> None:
    raw_dir = project.root / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fixture = raw_dir / "smoke_law.txt"
    if fixture.exists():
        return
    fixture.write_text(
        "\n".join(
            [
                "第一条 为保护劳动者合法权益，明确用人单位和劳动者的基本权利义务，制定本示例法。",
                "第二条 用人单位安排劳动者延长工作时间的，应当依法支付加班报酬并保障休息权。",
                "第三条 劳动者认为权益受到侵害的，可以保存劳动合同、考勤记录和工资凭证。",
                "第四条 争议发生后，当事人可以通过协商、调解、仲裁或者诉讼等合法路径解决。",
            ]
        ),
        encoding="utf-8",
    )


def prepare_p3_fixture(project: Project) -> None:
    from PIL import Image, ImageDraw

    data_dir = project.root / "data"
    images_dir = data_dir / "images"
    annotations_dir = data_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    image_items = []
    instance_annotations = []
    caption_annotations = []
    categories = [
        {"id": 1, "name": "chart"},
        {"id": 2, "name": "document"},
    ]
    for idx in range(1, 4):
        filename = f"{idx:012d}.jpg"
        path = images_dir / filename
        if not path.exists():
            image = Image.new("RGB", (640, 480), color=(245, 247, 250))
            draw = ImageDraw.Draw(image)
            draw.rectangle((80, 80, 560, 360), outline=(30, 64, 175), width=6)
            draw.text((120, 210), f"Smoke multimodal asset {idx}", fill=(20, 30, 45))
            image.save(path, quality=90)
        image_items.append({"id": idx, "file_name": filename, "width": 640, "height": 480})
        instance_annotations.append(
            {
                "id": idx,
                "image_id": idx,
                "category_id": 1 if idx % 2 else 2,
                "bbox": [80, 80, 480, 280],
                "area": 134400,
                "iscrowd": 0,
            }
        )
        caption_annotations.append(
            {
                "id": idx,
                "image_id": idx,
                "caption": f"A clean smoke-test image containing a framed document and chart element {idx}.",
            }
        )

    (annotations_dir / "instances_val2017.json").write_text(
        json.dumps({"images": image_items, "annotations": instance_annotations, "categories": categories}, indent=2),
        encoding="utf-8",
    )
    (annotations_dir / "captions_val2017.json").write_text(
        json.dumps({"images": image_items, "annotations": caption_annotations}, indent=2),
        encoding="utf-8",
    )


def gsm8k_records(count: int) -> list[dict]:
    records = []
    for idx in range(count):
        a = idx + 3
        b = idx + 5
        total = a + b
        records.append(
            {
                "question": f"Jamie has {a} apples and buys {b} more apples. How many apples does Jamie have now?",
                "answer": f"Jamie starts with {a} apples.\nJamie buys {b} more apples.\n{a}+{b}={total}.\n#### {total}",
            }
        )
    return records


def mbpp_records(count: int) -> list[dict]:
    records = []
    for idx in range(count):
        task_id = idx + 1
        records.append(
            {
                "task_id": task_id,
                "text": f"Write a Python function add_{task_id}(x) that returns x plus {task_id}.",
                "code": f"def add_{task_id}(x):\n    return x + {task_id}",
                "test_setup_code": "",
                "test_list": [f"assert add_{task_id}(1) == {task_id + 1}", f"assert add_{task_id}(3) == {task_id + 3}"],
                "challenge_test_list": [f"assert add_{task_id}(0) == {task_id}"],
            }
        )
    return records


def prepare_p4_fixture(project: Project) -> None:
    data_dir = project.root / "data"
    write_jsonl(data_dir / "gsm8k_train.jsonl", gsm8k_records(36))
    write_jsonl(data_dir / "mbpp_train.jsonl", mbpp_records(24))


def prepare_p5_fixture(project: Project) -> None:
    data_dir = project.root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    fixture = data_dir / "annual_report_2024_cn.txt"
    pages = []
    for page in range(1, 121):
        blocks = [
            f"第{page}页 华为2024年年度报告 smoke fixture。",
            "本页包含用于离线 RAG smoke test 的经营、研发、能源和网络建设信息。",
        ]
        if page in {1, 3, 16, 61}:
            blocks.append("2024年华为全年实现8,621亿销售收入，销售收入保持稳健。")
        if page in {1, 48, 50}:
            blocks.append("过去三年华为每年将销售收入的20%以上投入研究与开发。")
        if page == 2:
            blocks.append("目录里管理层讨论与分析在第12页开始。")
        if page == 19:
            blocks.append("在中国，分布式训练性能达到集中式训练性能的95%以上，分布式训练保持稳定。")
        if page == 20:
            blocks.append("在中东，AI辅助分析帮助运营商营销转化率提升超过10%。")
            blocks.append("在中国，机房改造后PUE从2.0优化到1.3。")
            blocks.append("在中国，智算数据中心总能耗降低约10%。")
        if page == 21:
            blocks.append("华为在撒哈拉沙漠边缘帮助建设了连续覆盖700公里的农网站点。")
        pages.append("\n\n".join(blocks))
    fixture.write_text("\f".join(pages), encoding="utf-8")


def prepare_smoke_fixtures(project: Project) -> None:
    if project.project_id == "P2":
        prepare_p2_fixture(project)
    elif project.project_id == "P3":
        prepare_p3_fixture(project)
    elif project.project_id == "P4":
        prepare_p4_fixture(project)
    elif project.project_id == "P5":
        prepare_p5_fixture(project)


def run_command(project: Project, script: str, timeout: int) -> dict:
    command = [sys.executable, script]
    env = os.environ.copy()
    env["PROJECT_SMOKE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(project.root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        duration = time.perf_counter() - started
        return {
            "script": script,
            "command": command,
            "duration_seconds": round(duration, 3),
            "returncode": completed.returncode,
            "passed": completed.returncode == 0,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - started
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            "script": script,
            "command": command,
            "duration_seconds": round(duration, 3),
            "returncode": None,
            "passed": False,
            "stdout": stdout.strip(),
            "stderr": (stderr.strip() + f"\nTimed out after {timeout} seconds.").strip(),
        }


def run_py_compile(project: Project, timeout: int) -> dict:
    src_files = sorted((project.root / "src").glob("*.py"))
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "py_compile", *[str(path) for path in src_files]],
        cwd=str(project.root),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return {
        "script": "py_compile src/*.py",
        "command": [sys.executable, "-m", "py_compile", "src/*.py"],
        "duration_seconds": round(time.perf_counter() - started, 3),
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def missing_required_files(project: Project) -> list[str]:
    return [
        rel(path)
        for path in [
            project.root / "README.md",
            project.root / "environment.yml",
            project.root / project.check_script,
        ]
        if not path.exists()
    ]


def run_project(project: Project, timeout: int, full: bool, clean: bool, no_prepare: bool) -> dict:
    missing = missing_required_files(project)
    if missing:
        return {
            "project": project.project_id,
            "passed": False,
            "mode": "full" if full else "quick",
            "removed_dirs": [],
            "steps": [],
            "stderr": "Missing required project files: " + ", ".join(missing),
        }

    removed_dirs = clean_generated_dirs(project) if clean and full else []
    if full and not no_prepare:
        prepare_smoke_fixtures(project)
    steps: list[dict] = []
    if not full:
        steps.append(run_py_compile(project, timeout))
    else:
        if not no_prepare:
            for script in project.prepare_scripts:
                step = run_command(project, script, timeout)
                steps.append(step)
                if not step["passed"]:
                    break
        if all(step["passed"] for step in steps):
            steps.append(run_command(project, project.check_script, timeout))

    return {
        "project": project.project_id,
        "passed": bool(steps) and all(step["passed"] for step in steps),
        "mode": "full" if full else "quick",
        "removed_dirs": removed_dirs,
        "steps": steps,
        "stderr": "",
    }


def render_markdown(results: list[dict], timestamp: str) -> str:
    passed = sum(1 for item in results if item["passed"])
    lines = [
        "# Project Smoke Test Report",
        "",
        f"- Timestamp: {timestamp}",
        f"- Overall status: {'PASS' if passed == len(results) else 'FAIL'}",
        f"- Passed projects: {passed}/{len(results)}",
        "",
        "| Project | Mode | Status | Steps | Duration |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        duration = sum(step.get("duration_seconds", 0) for step in result["steps"])
        lines.append(
            f"| {result['project']} | {result['mode']} | {status} | {len(result['steps'])} | {duration:.3f}s |"
        )

    lines.extend(["", "## Failure Details", ""])
    for result in results:
        if result["passed"]:
            continue
        lines.append(f"### {result['project']}")
        if result.get("stderr"):
            lines.extend(["", "```text", result["stderr"][-2000:], "```"])
        for step in result["steps"]:
            if step["passed"]:
                continue
            lines.append(f"- Failed step: `{step['script']}`")
            lines.append(f"- Return code: `{step['returncode']}`")
            if step["stderr"]:
                lines.extend(["", "```text", step["stderr"][-2000:], "```"])
            if step["stdout"]:
                lines.extend(["", "```text", step["stdout"][-2000:], "```"])
            break

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    selected = {item.upper() for item in args.project} if args.project else None
    projects = [item for item in PROJECTS if selected is None or item.project_id in selected]
    timestamp = datetime.now(timezone.utc).isoformat()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    results = [
        run_project(project, args.timeout, args.full, args.clean, args.no_prepare)
        for project in projects
    ]

    payload = {
        "timestamp_utc": timestamp,
        "overall_passed": all(item["passed"] for item in results),
        "results": results,
    }
    (args.report_dir / "project_smoke_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.report_dir / "project_smoke_report.md").write_text(render_markdown(results, timestamp), encoding="utf-8")

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{result['project']}: {status} ({result['mode']}, {len(result['steps'])} steps)")

    return 0 if payload["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
