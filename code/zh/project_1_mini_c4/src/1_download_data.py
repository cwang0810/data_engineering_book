import os

# ================= 配置部分 =================
# 1. 选择 Crawl ID (这是 2023 年第 50 周的抓取数据)
CRAWL_ID = "CC-MAIN-2023-50" 

# 2. 下载数量 (Mini项目建议 1-2 个，每个约 1GB)
NUM_FILES_TO_DOWNLOAD = 1

# 3. 数据保存目录
OUTPUT_DIR = "data/raw"

# Common Crawl 基础 URL
BASE_URL = "https://data.commoncrawl.org"
# ===========================================

def get_warc_file_paths(crawl_id, num_files):
    """
    获取指定 Crawl ID 的 WARC 文件下载路径列表
    """
    import gzip
    import requests

    # 路径索引文件地址
    paths_url = f"{BASE_URL}/crawl-data/{crawl_id}/warc.paths.gz"
    print(f"📡 正在获取文件索引: {paths_url} ...")
    
    try:
        # 流式下载索引文件
        response = requests.get(paths_url, stream=True, timeout=10)
        response.raise_for_status()
        
        paths = []
        # 解压并读取前 num_files 行
        with gzip.open(response.raw, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_files:
                    break
                paths.append(line.strip())
        return paths
    except Exception as e:
        print(f"❌ 获取索引失败: {e}")
        return []

def download_file(url, output_dir):
    """
    下载单个文件并显示进度条
    """
    import requests
    from tqdm import tqdm

    local_filename = url.split('/')[-1]
    local_path = os.path.join(output_dir, local_filename)
    
    if os.path.exists(local_path):
        print(f"⚠️ 文件已存在，跳过: {local_filename}")
        return local_path

    print(f"⬇️ 开始下载: {local_filename}")
    
    try:
        # stream=True 确保不会一次性把 1GB 读入内存
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(local_path, 'wb') as f, tqdm(
                desc=local_filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
        print(f"✅ 下载完成: {local_path}")
        return local_path
    except Exception as e:
        print(f"❌ 下载失败 {url}: {e}")
        # 如果下载失败，删除损坏的半成品文件
        if os.path.exists(local_path):
            os.remove(local_path)
        return None

def main():
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if os.environ.get("PROJECT_SMOKE") == "1":
        print("PROJECT_SMOKE=1，跳过 Common Crawl 下载，后续步骤会生成本地小样例。")
        return
    
    # 1. 获取文件路径列表
    warc_paths = get_warc_file_paths(CRAWL_ID, NUM_FILES_TO_DOWNLOAD)
    
    if not warc_paths:
        print("未找到文件路径，请检查网络或 CRAWL_ID。")
        return

    print(f"🎯 计划下载 {len(warc_paths)} 个文件到 {OUTPUT_DIR} ...")

    # 2. 循环下载
    for relative_path in warc_paths:
        full_url = f"{BASE_URL}/{relative_path}"
        download_file(full_url, OUTPUT_DIR)
    
    print("\n🎉 数据准备阶段完成！")

if __name__ == "__main__":
    main()
