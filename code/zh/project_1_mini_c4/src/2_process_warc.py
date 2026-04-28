import json
import os

# ================= 配置部分 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

WARC_FILENAME = "CC-MAIN-2023-50-segment-1700679099281.0-1700679117904.warc.gz"
INPUT_FILE = os.path.join(RAW_DIR, WARC_FILENAME)
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "extracted_data.jsonl")

# Mini 版本默认只处理前 10000 条记录，便于快速复现。
LIMIT_RECORDS = 10000
# ===========================================


def write_smoke_extracted_data(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    en_text = (
        "This smoke-test article explains how data engineers build reliable language model datasets. "
        "The pipeline collects source documents, normalizes text, removes duplicate records, checks quality, "
        "and prepares deterministic training and validation splits for repeatable experiments. "
        "Each paragraph is intentionally long enough to pass simple content filters and represent real prose. "
    )
    zh_text = (
        "这是一段用于烟测的数据工程中文样例 描述大模型语料流水线如何完成采集 清洗 去重 语言切分和质量过滤。"
        "它包含足够长度的自然语言内容 能够覆盖训练集构建 验证集拆分 质量统计和报告生成等关键步骤。"
        "这些记录只用于本地最小验证 不依赖外部网络 也不会下载真实大规模数据。"
    )
    records = []
    for index in range(36):
        records.append({"url": f"https://example.com/en/{index}", "text": en_text + f" English sample {index}."})
        records.append({"url": f"https://example.cn/zh/{index}", "text": zh_text + f" 中文样例{index}。"})
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"PROJECT_SMOKE=1，已写入本地小样例: {output_path}")


def extract_text_from_warc(warc_path, output_path, limit=None):
    """
    读取 WARC 文件，提取正文，并保存为 JSONL。
    """
    from tqdm import tqdm
    import trafilatura
    from warcio.archiveiterator import ArchiveIterator

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"🚀 开始处理: {warc_path}")
    print(f"💾 输出结果: {output_path}")

    counter = 0
    success_count = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        with open(warc_path, "rb") as stream:
            for record in tqdm(ArchiveIterator(stream), desc="Processing Records"):
                if record.rec_type != "response":
                    counter += 1
                    if limit and counter >= limit:
                        break
                    continue

                content_type = record.http_headers.get_header("Content-Type")
                if not content_type or "text/html" not in content_type:
                    counter += 1
                    if limit and counter >= limit:
                        break
                    continue

                try:
                    content = record.content_stream().read()
                except Exception:
                    counter += 1
                    if limit and counter >= limit:
                        break
                    continue

                text = trafilatura.extract(
                    content,
                    include_comments=False,
                    include_tables=False,
                    no_fallback=False,
                )

                if text and len(text.strip()) > 0:
                    url = record.rec_headers.get_header("WARC-Target-URI")
                    data = {
                        "url": url,
                        "text": text,
                    }
                    out_f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    success_count += 1

                counter += 1
                if limit and counter >= limit:
                    break

    print("\n✅ 处理完成！")
    print(f"📊 扫描记录数: {counter}")
    print(f"📄 成功提取数: {success_count}")


def main():
    if os.environ.get("PROJECT_SMOKE") == "1":
        write_smoke_extracted_data(OUTPUT_FILE)
        return

    input_path = INPUT_FILE
    if not os.path.exists(input_path):
        files = [f for f in os.listdir(RAW_DIR) if f.endswith(".warc.gz")]
        if files:
            input_path = os.path.join(RAW_DIR, files[0])
            print(f"自动发现文件: {input_path}")
        else:
            print(f"❌ 错误: 找不到输入文件 {INPUT_FILE}，且目录下没有其他 .warc.gz 文件")
            return

    extract_text_from_warc(input_path, OUTPUT_FILE, LIMIT_RECORDS)


if __name__ == "__main__":
    main()
