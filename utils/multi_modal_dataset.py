import os

import torch
from torch.utils.data import Dataset


class MultiModalVulDataset(Dataset):
    def __init__(self, vul_type, root='../dataset', train=True):
        self.vul_type = vul_type
        self.bytecode_token_dir = os.path.join(root, vul_type, "bytecode_tokens")
        self.cfg_graph_dir = os.path.join(root, vul_type, "cfg_pyg_data")
        self.source_vector_dir = os.path.join(root, vul_type, "source_vectors")
        self.names = self._load_names(os.path.join(root, vul_type, f"final_{vul_type}_name.txt"))
        self.labels = self._load_labels(os.path.join(root, vul_type, f"final_{vul_type}_label.txt"))
        self.train = train

    def _load_names(self, path):
        return [line.strip() for line in open(path)]

    def _load_labels(self, path):
        return [int(line.strip()) for line in open(path)]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx].replace('.sol', '')
        label = self.labels[idx]

        # 加载 token ids (序列特征)
        token_path = os.path.join(self.bytecode_token_dir, name + ".pt")
        bytecode_tokens = torch.load(token_path)  # [seq_len]

        # 加载图结构
        graph_path = os.path.join(self.cfg_graph_dir, name + ".pt")
        graph_data = torch.load(graph_path)  # PyG 的 Data 对象

        # 加载源码特征（仅训练阶段）
        if self.train:
            source_path = os.path.join(self.source_vector_dir, name + ".pt")
            source_vector = torch.load(source_path)  # [n_nodes, hidden_dim]
        else:
            source_vector = None

        return {
            'tokens': bytecode_tokens,
            'graph': graph_data,
            'source_vector': source_vector,
            'label': torch.tensor(label, dtype=torch.long)
        }


if __name__ == '__main__':
    dataset1 = MultiModalVulDataset('delegatecall')
