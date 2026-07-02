import os
import time

import torch.nn.functional as F

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import WeightedRandomSampler

# ====== 数据加载函数 ====== todo
from model.replace_conv1d_selfattention import MultiModeModel
from model.ours_tch import MultiModeModelTCH
from utils.draw import plot_training_curves
import random


def load_data(byte_data_dir, source_data_dir, graph_data_path, name_file, label_file):
    # 读取合约名称列表
    with open(name_file, 'r') as f:
        names = [line.strip() for line in f.readlines()]

    # 读取标签列表
    with open(label_file, 'r') as f:
        labels = [int(line.strip()) for line in f.readlines()]

    # 映射合约名到标签
    name2label = dict(zip(names, labels))

    data_list = []
    for file in list(set(os.listdir(byte_data_dir)) & set(os.listdir(source_data_dir))):
        if file.endswith(".pt"):
            # 加载bytecode
            file_path = os.path.join(byte_data_dir, file)
            tokens = torch.load(file_path)
            # 加载源码向量
            file_path = os.path.join(source_data_dir, file)
            vector = torch.load(file_path)
            # 加载图
            graph = torch.load(os.path.join(graph_data_path, file))

            contract_name = file.replace('.pt', '.sol')

            if contract_name in name2label:
                label = torch.tensor([name2label[contract_name]], dtype=torch.float)
                sample = {
                    "byte": tokens,
                    "source": vector,
                    "graph": graph,
                    "label": label
                }
                data_list.append(sample)
    return data_list


# 计算蒸馏损失
def compute_distill_loss(
        logits_student,
        logits_teacher,
        labels,
        criterion=nn.BCEWithLogitsLoss(),
        alpha=1.0,
        beta=0.5,
        temperature=2.0
):
    # === 1. Hard loss: 学生 vs 真实标签 ===
    loss_hard = criterion(logits_student, labels.float())

    # === 2. Soft loss: 学生 vs 老师 (soft labels)，用 KL 散度 ===
    # 注意 logits 需做 softmax + temperature
    p_student = F.log_softmax(logits_student / temperature, dim=-1)
    p_teacher = F.softmax(logits_teacher / temperature, dim=-1)
    loss_soft = F.kl_div(p_student, p_teacher, reduction='batchmean') * (temperature ** 2)
    # loss_soft = F.mse_loss(torch.sigmoid(logits_student), torch.sigmoid(logits_teacher))

    # === 3. 总损失 ===
    loss = alpha * loss_hard + beta * loss_soft
    return loss


# ====== 训练函数 ======
def train(model_std, loader, criterion, optimizer, device, model_tch):
    model_std.train()
    model_tch.eval()
    total_loss = 0

    y_true = []
    y_pred = []

    for batch_id, batch in enumerate(loader):
        # 拿数据
        data_byte = batch["byte"].to(device)
        data_source = batch["source"].to(device)
        data_graph = batch["graph"].to(device)
        labels = batch["label"].to(device).float().squeeze(1)  # 保证 shape = [batch_size]

        # teacher 使用 data 的拷贝（避免被 student 修改影响）todo
        data_graph_tch = data_graph.clone()  # PyG Data/Batch 支持
        data_byte_tch = data_byte.clone()  # 若是 tensor，也可 clone()

        # 学生模型训练
        logits_std = model_std(data_graph, data_byte)
        # 老师模型训练 todo
        with torch.no_grad():
            # logits_tch = model_tch(data_graph, data_byte, data_source)
            logits_tch = model_tch(data_graph_tch, data_byte_tch, data_source)

        # 计算蒸馏损失
        loss = compute_distill_loss(
            logits_student=logits_std,
            logits_teacher=logits_tch,
            labels=labels,
            criterion=criterion,
            alpha=1.0,
            beta=0.5,
            temperature=2.0
        )
        # 计算普通损失
        # loss=criterion(logits_std,labels.float())

        # 梯度下降，反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累加该batch总损失
        total_loss += loss.item() * data_byte.size(0)

        # 累计收集该batch的预测结果和真实结果
        preds = (torch.sigmoid(logits_std) > 0.5).int().cpu().tolist()
        labels_cpu = labels.int().cpu().tolist()
        y_true.extend(labels_cpu)
        y_pred.extend(preds)

        # 可选：打印 batch 信息
        # num_pos = sum(labels_cpu)
        # num_neg = len(labels_cpu) - num_pos
        # print(f"[Batch {batch_id}] Size: {len(labels_cpu)} | Positive: {num_pos} | Negative: {num_neg}")

    # 计算评估指标
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    # 计算平均损失
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, acc, precision, recall, f1


# ====== 测试函数 ======
def test(model_std, loader, criterion, device, model_tch):
    model_std.eval()
    model_tch.eval()
    total_loss = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            data_byte = batch["byte"].to(device)
            data_source = batch["source"].to(device)
            data_graph = batch["graph"].to(device)
            labels = batch["label"].to(device).float().squeeze(1)  # 保证 shape = [batch_size]

            # teacher 使用 data 的拷贝（避免被 student 修改影响）todo
            data_graph_tch = data_graph.clone()  # PyG Data/Batch 支持
            data_byte_tch = data_byte.clone()  # 若是 tensor，也可 clone()

            logits_std = model_std(data_graph, data_byte, data_source)
            # 老师模型训练 todo
            # logits_tch = model_tch(data_graph, data_byte, data_source)
            logits_tch = model_tch(data_graph_tch, data_byte_tch, data_source)

            # 计算蒸馏损失
            loss = compute_distill_loss(
                logits_student=logits_std,
                logits_teacher=logits_tch,
                labels=labels,
                criterion=criterion,
                alpha=1.0,
                beta=0.5,
                temperature=2.0
            )
            # 计算普通损失
            # loss = criterion(logits_std, labels.float())

            total_loss += loss.item() * data_byte.size(0)

            preds = (torch.sigmoid(logits_std) > 0.5).int().cpu().tolist()
            labels_cpu = labels.int().cpu().tolist()

            y_pred.extend(preds)
            y_true.extend(labels_cpu)

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

    return avg_loss, acc, precision, recall, f1


# 统计函数
def count_stats(dataset, name):
    total = len(dataset)
    pos = sum(int(d["label"].item() >= 0.5) for d in dataset)
    ratio = pos / total if total > 0 else 0
    print(f"{name} Set: Total={total}, Positives={pos}, Ratio={ratio:.2%}")


# todo
def get_balanced_sampler(train_list):
    labels = [int(data["label"].item()) for data in train_list]
    class_sample_count = [labels.count(0), labels.count(1)]
    weights = [1.0 / class_sample_count[label] for label in labels]

    # 固定随机种子
    generator = torch.Generator()
    generator.manual_seed(42)

    sampler = WeightedRandomSampler(weights, num_samples=len(train_list), replacement=True, generator=generator)

    # sampler = WeightedRandomSampler(weights, num_samples=len(train_list), replacement=True)
    return sampler

# todo
# torch.manual_seed(42)
# torch.cuda.manual_seed_all(42)
# random.seed(42)
# np.random.seed(42)
# torch.use_deterministic_algorithms(True)


# ====== 主训练入口 ======
def run_all_vulnerabilities(config_model,
                            config_model_tch,
                            vul_list,
                            base_path,
                            model_save_dir,
                            run_time,
                            epochs,
                            batch_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 获取当前秒级别时间
    time_str = time.strftime("%Y%m%d-%H%M%S")

    for vul in vul_list:
        # todo
        # if vul not in ["timestamp"]:
        #     continue

        num_runs = run_time  # 重复次数
        metrics_list = []  # 保存每次测试指标
        for run_id in range(num_runs):
            print(f"\n--- Training on Vulnerability: {vul} | Run {run_id + 1}/{num_runs} ---")

            byte_data_path = os.path.join(base_path, vul, "bytecode_tokens")
            source_data_path = os.path.join(base_path, vul, "source_vectors")
            graph_data_path = os.path.join(base_path, vul, "cfg_pyg_data")
            name_path = os.path.join(base_path, vul, f"final_{vul}_name.txt")
            label_path = os.path.join(base_path, vul, f"final_{vul}_label.txt")

            data_list = load_data(byte_data_path, source_data_path, graph_data_path, name_path, label_path)

            # ====== 打印漏洞分布统计 ======
            count_stats(data_list, vul)

            # ====== 划分训练和测试集 ======
            # 提取标签列表
            labels = [int(data["label"].item()) for data in data_list]

            # 使用 stratified split 保证正负样本比例
            train_list, test_list = train_test_split(
                data_list,
                test_size=0.3,
                stratify=labels,
                random_state=42,
            )

            # 打印统计信息
            count_stats(train_list, f"{vul} Train")
            count_stats(test_list, f"{vul} Test")

            # 权重采样
            train_loader = DataLoader(train_list, batch_size=batch_size, sampler=get_balanced_sampler(train_list))

            test_loader = DataLoader(test_list, batch_size=batch_size)

            # 学生模型
            model_std = MultiModeModel(**config_model).to(device)

            # 加载训练好的老师模型
            model_tch = MultiModeModelTCH(**config_model_tch).to(device)
            model_tch.load_state_dict(torch.load(f"../model_save/ours_tch_test/{vul}_best_model.pth"))

            criterion = nn.BCEWithLogitsLoss()
            # 1. 不加正则
            # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            # 加正则
            optimizer = torch.optim.Adam(model_std.parameters(), lr=1e-3, weight_decay=1e-4)
            # 学习率调度器
            scheduler = StepLR(optimizer, step_size=10, gamma=0.8)  # 每 10 个 epoch，lr 乘以 0.5

            train_losses, test_losses = [], []
            train_f1s, test_f1s = [], []
            train_precisions, test_precisions = [], []
            train_recalls, test_recalls = [], []

            best_f1 = 0
            patience = 20
            counter = 0

            for epoch in range(1, epochs + 1):
                train_loss, train_acc, train_prec, train_rec, train_f1 = train(model_std, train_loader, criterion,
                                                                               optimizer,
                                                                               device, model_tch)
                print(
                    f"Epoch {epoch:02d} Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | P: {train_prec:.4f} | R: {train_rec:.4f} | "
                    f"F1: {train_f1:.4f}")

                test_loss, test_acc, test_prec, test_rec, test_f1 = test(model_std, test_loader, criterion, device, model_tch)
                print(
                    f"****** Epoch {epoch:02d} Test  | Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | P: {test_prec:.4f} | R: {test_rec:.4f} | F1: {test_f1:.4f}")

                train_losses.append(train_loss)
                test_losses.append(test_loss)
                train_f1s.append(train_f1)
                test_f1s.append(test_f1)
                train_precisions.append(train_prec)
                test_precisions.append(test_prec)
                train_recalls.append(train_rec)
                test_recalls.append(test_rec)

                scheduler.step()  # 每个 epoch 更新一次学习率

                if test_f1 > best_f1:
                    best_f1 = test_f1
                    # todo
                    torch.save(model_std.state_dict(), f"../model_save/{model_save_dir}/conv1d/{vul}_{time_str}.pth")
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print("Early stopping triggered.")
                        break

            print(f"Finished training for {vul} Run {run_id + 1}.\n")
            plot_training_curves(
                train_losses, test_losses,
                train_f1s, test_f1s,
                train_precisions, test_precisions,
                train_recalls, test_recalls,
                vul=vul,
                # todo
                save_path=f'../plots/{model_save_dir}/conv1d/{vul}_{time_str}.png'
            )

            print(f"Start testing for {vul} Run {run_id + 1}.\n")
            # todo
            model_std.load_state_dict(torch.load(f"../model_save/{model_save_dir}/conv1d/{vul}_{time_str}.pth"))
            test_loss, test_acc, test_prec, test_rec, test_f1 = test(model_std, test_loader, criterion, device,model_tch)
            metrics_list.append({
                'loss': test_loss,
                'acc': test_acc,
                'prec': test_prec,
                'rec': test_rec,
                'f1': test_f1
            })
            print(
                f"Run {run_id + 1} Test  | Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | P: {test_prec:.4f} | R: {test_rec:.4f} | F1: {test_f1:.4f}")

        # 1. 打印原始 metrics_list
        print(f"原始 metrics_list（共 {len(metrics_list)} 项）:")
        for i, m in enumerate(metrics_list):
            print(
                f"  第{i + 1}组: Loss={m['loss']:.4f}, Acc={m['acc']:.4f}, P={m['prec']:.4f}, R={m['rec']:.4f}, F1={m['f1']:.4f}")

        # 2. 删除 F1 < 0.85 的组 todo
        # metrics_list = [m for m in metrics_list if m['f1'] >= 0.85]

        # 3. 打印过滤后的 metrics_list
        # print(f"\n过滤后 metrics_list（保留 {len(metrics_list)} 项，F1 >= 0.85）:")
        # for i, m in enumerate(metrics_list):
        #     print(
        #         f"  第{i + 1}组: Loss={m['loss']:.4f}, Acc={m['acc']:.4f}, P={m['prec']:.4f}, R={m['rec']:.4f}, F1={m['f1']:.4f}")

        # 4. 计算平均值
        avg_metrics = {
            'loss': np.mean([m['loss'] for m in metrics_list]),
            'acc': np.mean([m['acc'] for m in metrics_list]),
            'prec': np.mean([m['prec'] for m in metrics_list]),
            'rec': np.mean([m['rec'] for m in metrics_list]),
            'f1': np.mean([m['f1'] for m in metrics_list]),
        }

        # 5. 输出平均结果
        result_str = (f"{vul} Avg Test | Loss: {avg_metrics['loss']:.4f} | Acc: {avg_metrics['acc']:.4f} | "
                      f"P: {avg_metrics['prec']:.4f} | R: {avg_metrics['rec']:.4f} | F1: {avg_metrics['f1']:.4f}\n")

        print(result_str)

        # # 找出 F1 最大的那组指标
        # best_metric = max(metrics_list, key=lambda m: m['f1'])
        #
        # # 构造输出字符串
        # result_str = (f"{vul} Best Test | Loss: {best_metric['loss']:.4f} | Acc: {best_metric['acc']:.4f} | "
        #               f"P: {best_metric['prec']:.4f} | R: {best_metric['rec']:.4f} | F1: {best_metric['f1']:.4f}\n")

        # todo 写入文件
        with open(f"../model_save/{model_save_dir}/conv1d/result_conv1d_selfattention.txt", "a", encoding="utf-8") as f:
            f.write(result_str)



def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(42)

    config_model_std = {
        'vocab_size': 165,

        'hidden_dim_graph': 512,
        'net_type_graph': 'GGNN',
        'num_layers_graph': 2,
        'pool_type_graph': 'max',
        'dropout_rate_graph': 0.5,

        'hidden_dim_seq': 256,
        'net_type_seq': 'lstm',
        'num_layers_seq': 2,
        'pool_type_seq': 'max',
        'dropout_rate_seq': 0.5,

        'input_dim_source': 768,
        'num_layers_source': 2,
        'hidden_dim_source': 512,
        'dropout_rate_source': 0.5,
    }

    config_model_tch = {
        'vocab_size': 165,

        'hidden_dim_graph': 512,
        'net_type_graph': 'GGNN',
        'num_layers_graph': 2,
        'pool_type_graph': 'max',
        'dropout_rate_graph': 0.5,

        'hidden_dim_seq': 256,
        'net_type_seq': 'lstm',
        'num_layers_seq': 2,
        'pool_type_seq': 'max',
        'dropout_rate_seq': 0.5,

        'input_dim_source': 768,
        'num_layers_source': 2,
        'hidden_dim_source': 512,
        'dropout_rate_source': 0.5,
    }

    config_run = {
        'vul_list': ["reentrancy", "delegatecall", "timestamp", "integeroverflow"],
        'base_path': '../dataset',

        'model_save_dir': 'replace',
        'run_time': 5,
        'epochs': 100,
        'batch_size': 16
    }

    config_not_use = {
        'pool_type_fussion': 'max'
    }

    # 确保目录存在
    os.makedirs(f"../model_save/{config_run['model_save_dir']}", exist_ok=True)

    # 写日志文件（追加模式）todo
    log_path = f"../model_save/{config_run['model_save_dir']}/conv1d/result_conv1d_selfattention.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("==== Config ====\n")
        for key, value in {**config_run, **config_model_std, **config_not_use}.items():
            f.write(f"{key}: {value}\n")
        f.write("==== Results ====\n")

    run_all_vulnerabilities(config_model_std,config_model_tch, **config_run)
