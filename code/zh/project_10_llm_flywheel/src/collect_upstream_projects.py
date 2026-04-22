from __future__ import annotations

from collections import Counter

from pipeline_utils import BOOK_DIR, PROCESSED_DIR, ensure_standard_dirs, load_json, write_json

REGISTRY_FILE = PROCESSED_DIR / "upstream_project_registry.json"
PHASE_FILE = PROCESSED_DIR / "phase_inventory.json"

PROJECT_SPECS = [
    {
        "project_id": "p1",
        "title": "Mini-C4 Pretraining Corpus",
        "project_dir": "project_1_mini_c4",
        "metrics_file": "data/reports/p1_metrics.json",
        "test_file": "data/reports/p1_test_results.json",
        "phase": "acquisition",
        "deliverables": ["raw_corpus", "cleaned_corpus", "train_val_split"],
        "interfaces_out": ["foundation_corpus", "training_manifest"],
    },
    {
        "project_id": "p2",
        "title": "Legal SFT Factory",
        "project_dir": "project_2_sft_data",
        "metrics_file": "data/reports/p2_metrics.json",
        "test_file": "data/reports/p2_test_results.json",
        "phase": "alignment",
        "deliverables": ["domain_sft_dataset", "preference_pairs", "risk_refusals"],
        "interfaces_out": ["sft_corpus", "preference_data"],
    },
    {
        "project_id": "p3",
        "title": "LLaVA Multimodal Instruction Factory",
        "project_dir": "project_3_llava_data",
        "metrics_file": "data/reports/p3_metrics.json",
        "test_file": "data/reports/p3_test_results.json",
        "phase": "multimodal",
        "deliverables": ["image_text_pairs", "grounding_records", "interleaved_samples"],
        "interfaces_out": ["multimodal_sft"],
    },
    {
        "project_id": "p4",
        "title": "Synthetic Textbook Factory",
        "project_dir": "project_4_synth",
        "metrics_file": "data/reports/p4_metrics.json",
        "test_file": "data/reports/p4_test_results.json",
        "phase": "curriculum",
        "deliverables": ["verified_lessons", "books", "curriculum_map"],
        "interfaces_out": ["teaching_corpus", "verified_code_examples"],
    },
    {
        "project_id": "p5",
        "title": "Financial Report RAG",
        "project_dir": "project_5_rag",
        "metrics_file": "data/reports/p5_metrics.json",
        "test_file": "data/reports/p5_test_results.json",
        "phase": "application",
        "deliverables": ["rag_index", "eval_queries", "failure_replays"],
        "interfaces_out": ["knowledge_index", "retrieval_metrics"],
    },
    {
        "project_id": "p6",
        "title": "CoT + PRM Data Factory",
        "project_dir": "project_6_prm_data",
        "metrics_file": "data/reports/p6_metrics.json",
        "test_file": "data/reports/p6_test_results.json",
        "phase": "reasoning",
        "deliverables": ["cot_traces", "step_rewards", "prm_dataset"],
        "interfaces_out": ["prm_supervision", "reasoning_feedback"],
    },
    {
        "project_id": "p7",
        "title": "Agent Tool-Use Factory",
        "project_dir": "project_7_agent_tooluse",
        "metrics_file": "data/reports/p7_metrics.json",
        "test_file": "data/reports/p7_test_results.json",
        "phase": "agent",
        "deliverables": ["tool_schemas", "agent_trajectories", "safety_refusals"],
        "interfaces_out": ["agent_training_data", "tooling_specs"],
    },
    {
        "project_id": "p8",
        "title": "Enterprise DataOps Platform",
        "project_dir": "project_8_dataops_platform",
        "metrics_file": "data/reports/p8_metrics.json",
        "test_file": "data/reports/p8_test_results.json",
        "phase": "platform",
        "deliverables": ["dataset_registry", "lineage_graph", "ops_policies"],
        "interfaces_out": ["platform_control_plane", "ops_metrics"],
    },
    {
        "project_id": "p9",
        "title": "Privacy-Preserving Pipeline",
        "project_dir": "project_9_privacy_pipeline",
        "metrics_file": "data/reports/p9_metrics.json",
        "test_file": "data/reports/p9_test_results.json",
        "phase": "privacy",
        "deliverables": ["classification_policy", "redacted_records", "incident_playbook"],
        "interfaces_out": ["privacy_controls", "sanitized_exports"],
    },
]


def extract_kpis(project_id: str, metrics: dict) -> dict:
    if project_id == "p1":
        return {
            "records": metrics["final_summary"]["doc_count"],
            "tokens": metrics["final_summary"]["estimated_tokens"],
            "quality_retention": metrics["retention"]["final_over_extracted"],
        }
    if project_id == "p2":
        return {
            "records": metrics["final_dataset_count"],
            "preference_pairs": metrics["preference_pair_count"],
            "review_score": metrics["average_review_score"],
        }
    if project_id == "p3":
        return {
            "records": metrics["training_manifest"]["num_records"],
            "qa_pass_rate": metrics["quality_pass_rate"],
            "assets": metrics["num_assets_total"],
        }
    if project_id == "p4":
        return {
            "records": metrics["training_manifest"]["num_records"],
            "verification_pass_rate": metrics["verification_pass_rate"],
            "books": metrics["num_books"],
        }
    if project_id == "p5":
        return {
            "queries": metrics["num_queries"],
            "retrieval_hit_rate": metrics["retrieval_hit_rate_at_4"],
            "citation_accuracy": metrics["citation_accuracy"],
        }
    if project_id == "p6":
        return {
            "step_records": metrics["training_manifest"]["num_records"],
            "trace_count": metrics["trace_count"],
            "process_signal_steps": metrics["process_supervision_only_signal_steps"],
        }
    if project_id == "p7":
        return {
            "trajectory_count": metrics["trajectory_count"],
            "trajectory_success_rate": metrics["trajectory_success_rate"],
            "memory_success_rate": metrics["memory_success_rate"],
        }
    if project_id == "p8":
        return {
            "dataset_versions": metrics["dataset_version_count"],
            "experiments": metrics["experiment_count"],
            "sla_met_rate": metrics["sla_met_rate"],
        }
    return {
        "restricted_records": metrics["restricted_record_count"],
        "direct_pii_removed_rate": metrics["direct_pii_removed_rate"],
        "preflight_pass_rate": metrics["preflight_pass_rate"],
    }


def main() -> None:
    ensure_standard_dirs()
    registry: list[dict] = []
    for spec in PROJECT_SPECS:
        project_root = BOOK_DIR / spec["project_dir"]
        metrics = load_json(project_root / spec["metrics_file"])
        tests = load_json(project_root / spec["test_file"])
        registry.append(
            {
                **spec,
                "metrics_path": str(project_root / spec["metrics_file"]),
                "tests_path": str(project_root / spec["test_file"]),
                "overall_passed": tests["overall_passed"],
                "passed_checks": tests["passed_checks"],
                "total_checks": tests["total_checks"],
                "kpis": extract_kpis(spec["project_id"], metrics),
                "estimated_manual_review_hours": metrics.get("estimated_manual_review_hours", 0.0),
                "estimated_manual_review_cost_rmb": metrics.get("estimated_manual_review_cost_rmb", 0.0),
            }
        )

    phase_inventory = {
        "project_count": len(registry),
        "phase_distribution": dict(Counter(item["phase"] for item in registry)),
        "all_projects_passed": all(item["overall_passed"] for item in registry),
        "total_passed_checks": sum(item["passed_checks"] for item in registry),
        "total_checks": sum(item["total_checks"] for item in registry),
        "interface_count": sum(len(item["interfaces_out"]) for item in registry),
    }

    write_json(registry, REGISTRY_FILE)
    write_json(phase_inventory, PHASE_FILE)
    print("✅ P10 上游项目资产采集完成。")
    print(phase_inventory)


if __name__ == "__main__":
    main()
