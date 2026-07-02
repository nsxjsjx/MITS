import os
import torch
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm


def extract_features(source_code,tokenizer,MAX_LENGTH,device,model):
    """使用GraphCodeBERT提取源码特征向量"""
    try:
        # Tokenize并处理长代码截断
        inputs = tokenizer(
            source_code,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(device)

        # 前向传播
        with torch.no_grad():
            outputs = model(**inputs)

        # 取[CLS]位置的向量作为整体表征
        cls_vector = outputs.last_hidden_state[:, 0, :].cpu()
        return cls_vector.squeeze(0)  # 从(1, 768)变为(768,)
    except Exception as e:
        print(f"Error processing code: {str(e)}")
        return None


def sourcecode_to_bert():
    # 配置参数
    DATASET_ROOT = "../dataset"
    MODEL_NAME = "microsoft/graphcodebert-base"
    MAX_LENGTH = 512  # GraphCodeBERT最大输入长度

    # 加载GraphCodeBERT模型和tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaModel.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    """遍历所有漏洞类型目录处理合约"""
    for vul in os.listdir(DATASET_ROOT):
        source_dir = os.path.join(DATASET_ROOT, vul, "sourcecode")
        output_dir = os.path.join(DATASET_ROOT, vul, "source_vectors")

        # 保证这是一个文件夹
        if not os.path.isdir(source_dir):
            continue

        # 创建输出文件夹
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nProcessing {vul}...")

        # 处理该漏洞的合约
        for filename in tqdm(os.listdir(source_dir)):
            if not filename.endswith(".sol"):
                continue

            # 读取源码
            filepath = os.path.join(source_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                code = f.read()

            # 提取特征
            vector = extract_features(code,tokenizer,MAX_LENGTH,device,model)
            if vector is None:
                continue

            # 保存为.pt格式
            base_name = os.path.splitext(filename)[0]
            torch.save(vector, os.path.join(output_dir, f"{base_name}.pt"))


if __name__ == "__main__":
    sourcecode_to_bert()
    print("\nAll features extracted!")
