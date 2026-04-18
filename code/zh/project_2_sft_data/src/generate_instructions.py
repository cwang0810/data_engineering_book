from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pipeline_utils import (
    estimated_tokens,
    load_jsonl,
    normalize_text,
    processed_dir,
    trim_summary,
    write_jsonl,
)


PROCESSED_DIR = processed_dir()
SEED_FILE = PROCESSED_DIR / "legal_seed_dataset.jsonl"
OUTPUT_FILE = PROCESSED_DIR / "domain_expert_sft.jsonl"
REJECTED_FILE = PROCESSED_DIR / "synthetic_candidates_rejected.jsonl"

TEACHER_NAME = "template_teacher_v2"
JUDGE_NAME = "heuristic_legal_judge_v1"


CASE_SCENARIOS = {
    "中华人民共和国劳动法": "某员工连续三个月被安排超时加班，单位只按基本工资发放报酬，员工想主张加班费和解除劳动合同补偿。",
    "中华人民共和国刑法": "某人将他人银行卡信息偷偷绑定到自己的支付工具并小额多次转账，涉案金额累计较大。",
    "中华人民共和国公司法": "公司大股东未经股东会程序决定对外担保，小股东担心公司承担过高风险并质疑决议效力。",
    "中华人民共和国民事诉讼法": "原告在借款纠纷中只保留了转账记录，没有正式借条，被告否认借款关系并提出管辖异议。",
    "中华人民共和国民法典": "购房人收房后发现房屋存在渗水和面积误差，开发商以合同免责条款为由拒绝承担责任。",
}


def explain_task(seed: dict) -> tuple[str, str]:
    law_name = seed["law_name"]
    article_no = seed["article_no"]
    content = seed["content"]
    instruction = (
        f"请用通俗、专业且合规的方式解释《{law_name}》{article_no}。"
        f" 需要说明该条文主要规范什么、普通人遇到什么场景会用到它。"
    )
    output = f"""#### 问题重述
用户想理解《{law_name}》{article_no}的核心含义和实际用途。

#### 法律依据
{content}

#### 结论与建议
这条规定的核心在于：{trim_summary(content, 48)}。如果普通人遇到与该条文对应的争议场景，应先确认事实、保存证据，再判断自己是否属于该条文调整的对象；如争议已经进入诉讼或合同履行阶段，建议结合完整法条体系和具体证据进一步分析。"""
    return instruction, output


def qa_task(seed: dict) -> tuple[str, str]:
    law_name = seed["law_name"]
    article_no = seed["article_no"]
    content = seed["content"]
    instruction = (
        f"围绕《{law_name}》{article_no}，回答一个法律咨询问题："
        f"如果当事人怀疑自己的权益被侵害，应当如何理解并使用这一条文？"
    )
    output = f"""#### 问题重述
当事人希望知道《{law_name}》{article_no}能否作为自己主张权利的依据。

#### 法律依据
{content}

#### 结论与建议
可以把这条法条理解为一个“判断框架”，先看自己的事实是否落入该条文描述的情形，再看是否有证据支持。如果事实匹配，就可以把它作为沟通、投诉、协商或诉讼中的法律依据之一；如果事实不完全匹配，则需要结合上下位条文或相关司法解释一并判断，避免只凭单一条文下结论。"""
    return instruction, output


def case_task(seed: dict) -> tuple[str, str]:
    law_name = seed["law_name"]
    article_no = seed["article_no"]
    content = seed["content"]
    scenario = CASE_SCENARIOS.get(law_name, "当事人因合同履行、权利义务边界或程序选择发生争议，希望判断应如何依法处理。")
    instruction = (
        f"请结合《{law_name}》{article_no}分析案例：{scenario}"
        " 要求给出争议焦点、法条适用和可执行建议。"
    )
    output = f"""#### 事实识别
本案需要先确认关键事实是否真实、证据是否完整，以及争议行为发生的时间、主体和法律关系。

#### 争议焦点
1. 争议事实是否落入《{law_name}》{article_no}的调整范围。
2. 当事人提交的证据能否支撑其主张。
3. 除本条之外，是否还需要结合相关配套条文或程序规定。

#### 法律适用
《{law_name}》{article_no}规定：{content}
因此，在本案中应先把争议事实与条文要件逐项对应，再判断权利义务和责任分配。

#### 行动建议
建议当事人先固定证据、梳理时间线，并围绕该条文对应的要件组织主张。若协商无法解决，可进一步选择投诉、仲裁或诉讼路径，但在正式行动前应结合完整案情做更细致的法律分析。"""
    return instruction, output


TASK_BUILDERS = {
    "legal_qa": qa_task,
    "statute_explanation": explain_task,
    "case_analysis": case_task,
}


def judge_candidate(item: dict) -> tuple[float, list[str]]:
    reasons = []
    score = 0.0
    output = item["output"]
    if len(item["instruction"]) >= 30:
        score += 0.2
    else:
        reasons.append("instruction_too_short")
    if "法律依据" in output or "法律适用" in output:
        score += 0.2
    else:
        reasons.append("missing_legal_basis")
    if "建议" in output or "结论" in output:
        score += 0.2
    else:
        reasons.append("missing_actionable_advice")
    if item["law_name"] in output and item["article_no"] in output:
        score += 0.2
    else:
        reasons.append("missing_citation")
    if estimated_tokens(output) >= 120:
        score += 0.2
    else:
        reasons.append("too_short_output")
    return round(score, 3), reasons


def make_rejected_variant(item: dict) -> dict:
    rejected_output = (
        "#### 结论\n"
        "建议直接按自己的理解处理，不需要进一步核对事实，也不需要查看相关证据或其他法条。"
    )
    return {
        "sample_id": item["sample_id"],
        "seed_id": item["seed_id"],
        "task_type": item["task_type"],
        "instruction": item["instruction"],
        "input": "",
        "output": rejected_output,
        "domain": "legal",
        "law_name": item["law_name"],
        "article_no": item["article_no"],
        "source_doc": item["source_doc"],
        "teacher_model": "weak_baseline_v1",
        "judge_model": JUDGE_NAME,
        "judge_score": 0.12,
        "judge_reasons": ["missing_citation", "unsafe_overclaim", "too_short_output"],
        "status": "rejected",
    }


def main() -> None:
    if not SEED_FILE.exists():
        raise FileNotFoundError(f"Missing seed file: {SEED_FILE}")

    seeds = load_jsonl(SEED_FILE)
    accepted = []
    rejected = []

    for seed in seeds:
        for task_type, builder in TASK_BUILDERS.items():
            instruction, output = builder(seed)
            sample_id = f"{seed['id']}::{task_type}"
            candidate = {
                "sample_id": sample_id,
                "seed_id": seed["id"],
                "instruction": normalize_text(instruction),
                "input": "",
                "output": normalize_text(output),
                "task_type": task_type,
                "source_doc": seed["source"],
                "law_name": seed["law_name"],
                "article_no": seed["article_no"],
                "domain": "legal",
                "teacher_model": TEACHER_NAME,
                "judge_model": JUDGE_NAME,
            }
            score, reasons = judge_candidate(candidate)
            candidate["judge_score"] = score
            candidate["judge_reasons"] = reasons
            candidate["status"] = "accepted" if score >= 0.8 else "rejected"

            if candidate["status"] == "accepted":
                accepted.append(candidate)
                rejected.append(make_rejected_variant(candidate))
            else:
                rejected.append(candidate)

    write_jsonl(OUTPUT_FILE, accepted)
    write_jsonl(REJECTED_FILE, rejected)

    task_distribution = Counter(item["task_type"] for item in accepted)
    print("✅ 法律 SFT 合成完成。")
    print(f"Accepted: {len(accepted)}")
    print(f"Rejected: {len(rejected)}")
    print(json.dumps({"task_distribution": dict(task_distribution)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
