import json
import os

from pipeline_utils import sha1_text

# ================= 配置 =================
# 自动设置路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

INPUT_FILE = os.path.join(DATA_DIR, "clean_data.jsonl")  # 上一步清洗完的文件
OUTPUT_FILE = os.path.join(DATA_DIR, "deduplicated_data.jsonl")

# MinHash 参数 (C4 标准参数: num_perm=128)
NUM_PERM = 128
THRESHOLD = 0.8  # 相似度阈值，超过 0.8 视为重复


def run_smoke_dedup():
    seen = set()
    unique_records = []
    duplicate_count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            text_hash = sha1_text(item.get("text", ""))
            if text_hash in seen:
                duplicate_count += 1
                continue
            seen.add(text_hash)
            unique_records.append(item)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in unique_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("PROJECT_SMOKE=1，使用精确哈希去重。")
    print(f"发现重复: {duplicate_count}")
    print(f"剩余有效: {len(unique_records)}")


def run_minhash_dedup():
    from datasketch import MinHash, MinHashLSH
    from tqdm import tqdm
    import ray

    ray.init(ignore_reinit_error=True)

    def get_minhash(text, num_perm=128):
        m = MinHash(num_perm=num_perm)
        words = text.split()
        for w in words:
            m.update(w.encode("utf8"))
        return m

    @ray.remote
    def process_batch(lines, batch_id):
        results = []
        for line in lines:
            try:
                item = json.loads(line)
                url = item["url"]
                text = item["text"]
                minhash = get_minhash(text, NUM_PERM)
                results.append((url, minhash, text))
            except Exception:
                continue
        return results

    print("🚀 第一阶段: 并行计算 MinHash 签名...")

    batch_size = 1000
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    total_records = len(all_lines)
    print(f"📚 总记录数: {total_records}")
    batches = [all_lines[i:i + batch_size] for i in range(0, total_records, batch_size)]
    futures = [process_batch.remote(batch, i) for i, batch in enumerate(batches)]

    print("⏳ 等待 CPU 计算中 ")
    processed_batches = ray.get(futures)
    results = [item for batch in processed_batches for item in batch]

    print("\n🚀 第二阶段: 构建 LSH 索引并去重...")
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    unique_records = []
    duplicate_count = 0

    for url, minhash, text in tqdm(results, desc="LSH Deduplication"):
        duplicates = lsh.query(minhash)
        if len(duplicates) > 0:
            duplicate_count += 1
        else:
            lsh.insert(url, minhash)
            unique_records.append({"url": url, "text": text})

    print("\n✅ 去重完成！")
    print(f"🗑️ 发现重复: {duplicate_count}")
    print(f"💎 剩余有效: {len(unique_records)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in unique_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    ray.shutdown()


# ================= 主流程 =================
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件: {INPUT_FILE}")
        return

    if os.environ.get("PROJECT_SMOKE") == "1":
        run_smoke_dedup()
    else:
        run_minhash_dedup()


if __name__ == "__main__":
    main()
