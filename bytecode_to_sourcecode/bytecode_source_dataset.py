import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

from utils.bytecode_to_token import BytecodeOrOpcodeTokenizer


class BytecodeSourceDataset(Dataset):
    def __init__(self, root_dir: str = "../dataset"):
        """
        初始化数据集
        Args:
            root_dir: 数据集根目录，结构应为：
                ../dataset/
                ├── {vul_type}/
                │   ├── bytecode/       # 含.sol文件（含Binary字节码）
                │   └── source_vectors/ # 含.pt特征文件
        """
        self.pairs = []

        self.byte_tokenizer = BytecodeOrOpcodeTokenizer()
        self.tokens_len=self.byte_tokenizer.max_len
        self.vocab=self.byte_tokenizer.vocab

        # 收集所有有效的字节码-源码向量对
        for vul_type in os.listdir(root_dir):
            bytecode_dir = os.path.join(root_dir, vul_type, "bytecode_pure")
            vector_dir = os.path.join(root_dir, vul_type, "source_vectors")

            if not (os.path.isdir(bytecode_dir) and os.path.isdir(vector_dir)):
                continue

            # 构建文件名映射（不含扩展名）
            bytecode_files = {
                os.path.splitext(f)[0]: f
                for f in os.listdir(bytecode_dir)
                if f.endswith('.sol')
            }
            vector_files = {
                os.path.splitext(f)[0]: f
                for f in os.listdir(vector_dir)
                if f.endswith('.pt')
            }

            # 匹配成对的文件
            common_names = set(bytecode_files.keys()) & set(vector_files.keys())
            for name in common_names:
                self.pairs.append((
                    os.path.join(bytecode_dir, bytecode_files[name]),
                    os.path.join(vector_dir, vector_files[name])
                ))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
            bytecode_tensor: 形状为 [seq_len] 的字节码token序列
            source_vector: 形状为 [768] 的GraphCodeBERT特征向量
        """
        bytecode_path, vector_path = self.pairs[idx]

        # 1. 加载源码特征向量
        try:
            source_vector = torch.load(vector_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load vector {vector_path}: {str(e)}")

        # 2. 解析字节码文件
        with open(bytecode_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 3. 转换为token序列
        bytecode_tensor, attention_mask = self.byte_tokenizer.tokenize_bytecode(content)
        return bytecode_tensor, attention_mask, source_vector

    @staticmethod
    def collate_fn(batch):
        """处理批次数据（过滤无效样本并堆叠）"""
        valid_batch = [item for item in batch if item is not None]
        if not valid_batch:
            return None, None

        bytecode_tensors, attention_mask,source_vectors = zip(*valid_batch)
        return torch.stack(bytecode_tensors), torch.stack(attention_mask),torch.stack(source_vectors)


def main():
    dataset = BytecodeSourceDataset("../dataset")
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=dataset.collate_fn,
        shuffle=True
    )

    for bytecode_batch, attention_mask, source_vecs in dataloader:
        if bytecode_batch is None:  # 跳过无效批次
            continue
        # bytecode_batch: [batch_size, seq_len]
        # source_vecs: [batch_size, 768]
        # ...训练逻辑...


if __name__ == '__main__':
    main()
