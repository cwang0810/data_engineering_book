from datasets import load_dataset
import json

def save_to_jsonl(dataset, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"已成功保存至: {filename}")

# --- 1. 下载 GSM8K (数学推理) ---
print("正在下载 GSM8K...")
# main 包含 'train' 和 'test' 分组
gsm8k = load_dataset("openai/gsm8k", "main")
save_to_jsonl(gsm8k['train'], 'gsm8k_train.jsonl')

# --- 2. 下载 MBPP (代码生成) ---
print("正在下载 MBPP...")
# full 包含原始的训练数据
mbpp = load_dataset("google-research-datasets/mbpp", "full")
save_to_jsonl(mbpp['train'], 'mbpp_train.jsonl')