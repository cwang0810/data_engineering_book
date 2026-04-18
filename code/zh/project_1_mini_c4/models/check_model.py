import kenlm
import os

model_path = "/home/xuxin123/book/project_1_mini_c4/models/en.arpa.bin"

if not os.path.exists(model_path):
    print("❌ 文件不存在！")
else:
    try:
        # 尝试加载模型
        model = kenlm.Model(model_path)
        print("✅ 模型加载成功！文件是完整的。")
        
        # 尝试跑一个简单的分
        test_text = "This is a test sentence to check if the model works."
        score = model.score(test_text)
        print(f"测试得分: {score}")
        
    except Exception as e:
        print(f"❌ 模型损坏或不完整！错误信息: {e}")