import os

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import WeightedRandomSampler
from torch_geometric.data import Batch
from model.ours import FusionModel
from utils.bytecode_to_token import BytecodeOrOpcodeTokenizer
from utils.multi_modal_dataset import MultiModalVulDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


def custom_collate_fn(batch):
    """
    batch: List[dict]，每个 dict 形如：
        {
            'tokens': tensor(seq_len),
            'graph': PyG Data,
            'source_vector': tensor([n_nodes, hidden_dim]) 或 None,
            'label': tensor(1)
        }
    """

    tokens_list = [item['tokens'] for item in batch]
    token_padded = pad_sequence(tokens_list, batch_first=True)  # [B, max_seq_len]

    graphs = Batch.from_data_list([item['graph'] for item in batch])  # 合并 PyG 图

    # 处理 source_vector（训练阶段可能有，测试时为 None）
    if batch[0]['source_vector'] is not None:
        source_vectors = [item['source_vector'] for item in batch]
        source_padded = pad_sequence(source_vectors, batch_first=True)  # [B, max_n_nodes, hidden_dim]
    else:
        source_padded = None

    labels = torch.stack([item['label'] for item in batch])  # [B]

    return {
        'tokens': token_padded,  # [B, max_seq_len]
        'graph': graphs,  # PyG Batch 对象
        'source_vector': source_padded,  # [B, max_nodes, hidden_dim] or None
        'label': labels  # [B]
    }


def build_dataloaders(vul_type, batch_size=16):
    # 1. 加载完整数据集
    full_dataset = MultiModalVulDataset(vul_type)
    # 2. 获取标签
    all_labels = [sample['label'] for sample in full_dataset]
    # 3. 首先分出测试集（20%）
    train_val_idx, test_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        stratify=all_labels,
        random_state=42
    )
    # 4. 再从 train_val 中划分验证集（1/8 of train_val）
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.125,
        stratify=[all_labels[i] for i in train_val_idx],
        random_state=42
    )
    # 5. 构造子集
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    # 加权随机采样器
    def get_balanced_sampler(subset):
        labels = [subset.dataset[i]['label'] for i in subset.indices]
        class_sample_count = np.array([len(np.where(np.array(labels) == t)[0]) for t in [0, 1]])
        # 标签少的类别可以分配更高的权重，从而增加被采样的概率
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in labels])
        # replacement是否允许重复采样（True 表示放回采样）
        sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)
        return sampler

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=get_balanced_sampler(train_dataset),
                              collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    return train_loader, val_loader, test_loader


def train_model(vul_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = build_dataloaders(vul_type)

    model = FusionModel(BytecodeOrOpcodeTokenizer().vocab_size, 512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce_loss_fn = nn.BCEWithLogitsLoss()  # 内部自动加了 Sigmoid
    mse_loss_fn = nn.MSELoss()

    patience = 15
    best_f1 = -1

    # === 训练阶段 ===
    for epoch in range(100):
        model.train()
        total_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []

        for batch in train_loader:
            tokens = batch['tokens'].to(device)  # [B, L]
            graphs = batch['graph'].to(device)  # PyG Batch
            source_vecs = batch['source_vector'].to(device)  # [B, D]
            labels = batch['label'].to(device).float()  # [B]

            out, source_info_learn = model(tokens, graphs, source_vecs)  # out: [B, 1]

            out = out.squeeze(1)  # [B]

            # 主任务 loss
            main_loss = bce_loss_fn(out, labels)

            # 对齐源代码向量和预测源向量
            align_loss = mse_loss_fn(source_info_learn, source_vecs)

            loss = main_loss + align_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # ====== 收集用于评估 ======
            probs = torch.sigmoid(out).detach().cpu()
            preds = (probs > 0.5).long()
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().tolist())

        # ====== 指标计算 ======
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0  # 某一类全为0会报错

        print(
            f"[{vul_type}] Epoch {epoch}: Loss {total_loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        # === 验证阶段 ===
        model.eval()
        val_preds = []
        val_labels = []
        val_probs = []

        with torch.no_grad():
            for batch in val_loader:
                tokens = batch['tokens'].to(device)
                graphs = batch['graph'].to(device)
                source_vecs = batch['source_vector'].to(device)
                labels = batch['label'].to(device).float()

                out, _ = model(tokens, graphs, source_vecs)
                out = out.squeeze(1)
                probs = torch.sigmoid(out).cpu()
                preds = (probs > 0.5).long()

                val_preds.extend(preds.tolist())
                val_probs.extend(probs.tolist())
                val_labels.extend(labels.cpu().tolist())

        # === 计算指标 ===
        val_acc = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds)
        val_recall = recall_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_probs)

        print(f"[{vul_type}] Epoch {epoch}: "
              f"Acc = {val_acc:.4f}, "
              f"Prec = {val_precision:.4f}, "
              f"Recall = {val_recall:.4f}, "
              f"F1 = {val_f1:.4f}, "
              f"AUC = {val_auc:.4f}")

        # === 提前停止判断（基于 F1）===
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), f'../model_save/best_model_{vul_type}.pt')  # 可选保存模型
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # === 测试阶段 ===
    # === 加载保存的最佳模型参数 ===
    model.load_state_dict(torch.load(f'../model_save/best_model_{vul_type}.pt'))
    model.to(device)
    model.eval()
    test_preds = []
    test_labels = []
    test_probs = []

    with torch.no_grad():
        for batch in test_loader:
            tokens = batch['tokens'].to(device)
            graphs = batch['graph'].to(device)
            source_vecs = batch['source_vector'].to(device)
            labels = batch['label'].to(device).float()

            out, _ = model(tokens, graphs, source_vecs)
            out = out.squeeze(1)
            probs = torch.sigmoid(out).cpu()
            preds = (probs > 0.5).long()

            test_preds.extend(preds.tolist())
            test_probs.extend(probs.tolist())
            test_labels.extend(labels.cpu().tolist())

    # === 计算指标 ===
    test_acc = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds)
    test_recall = recall_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_probs)

    print(f"[*] Test Results:")
    print(f"Accuracy  : {test_acc:.4f}")
    print(f"Precision : {test_precision:.4f}")
    print(f"Recall    : {test_recall:.4f}")
    print(f"F1 Score  : {test_f1:.4f}")
    print(f"AUC       : {test_auc:.4f}")


if __name__ == '__main__':
    for vul in os.listdir('../dataset'):
        if os.listdir(f'../dataset/{vul}'):
            train_model(vul)
