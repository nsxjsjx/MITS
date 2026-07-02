import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import WeightedRandomSampler

# ====== 数据加载函数 ======
from model.only_graph import GraphModel
from utils.draw import plot_training_curves


def load_data(data_dir, name_file, label_file):
    # 读取标签
    with open(name_file, 'r') as f:
        names = [line.strip() for line in f.readlines()]
    with open(label_file, 'r') as f:
        labels = [int(line.strip()) for line in f.readlines()]
    name2label = dict(zip(names, labels))

    # 加载目录下所有 .pt 文件
    graph_list = []
    for file in os.listdir(data_dir):
        if file.endswith(".pt"):
            data = torch.load(os.path.join(data_dir, file))
            name = file.replace('.pt', '.sol')
            if name in name2label:
                data.y = torch.tensor([name2label[name]], dtype=torch.float)
                graph_list.append(data)
    return graph_list


# ====== 训练函数 ======
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    y_true = []
    y_pred = []

    for batch_id, data in enumerate(loader):
        # y = data.y.cpu().numpy()
        # num_pos = (y == 1).sum()
        # num_neg = (y == 0).sum()
        # print(f"[Batch {batch_id}] Size: {len(y)} | Positive: {num_pos} | Negative: {num_neg}")

        data = data.to(device)

        out = model(data).squeeze()
        loss = criterion(out, data.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(data)

        # 收集标签和预测
        preds = (torch.sigmoid(out) > 0.5).int().cpu().tolist()
        labels = data.y.int().cpu().tolist()

        y_true.extend(labels)
        y_pred.extend(preds)

    # 计算评估指标
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, acc, precision, recall, f1


# ====== 测试函数 ======
def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data).squeeze()
            loss = criterion(out, data.y)
            total_loss += loss.item() * len(data)
            preds = torch.sigmoid(out) > 0.5
            y_pred.extend(preds.cpu().int().tolist())
            y_true.extend(data.y.cpu().int().tolist())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return avg_loss, acc, precision, recall, f1


# 统计函数
def count_stats(dataset, name):
    total = len(dataset)
    pos = sum(int(d.y.item() == 1) for d in dataset)
    ratio = pos / total if total > 0 else 0
    print(f"{name} Set: Total={total}, Positives={pos}, Ratio={ratio:.2%}")


def get_balanced_sampler(train_list):
    labels = [int(data.y.item()) for data in train_list]
    class_sample_count = [labels.count(0), labels.count(1)]
    weights = [1.0 / class_sample_count[label] for label in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_list), replacement=True)
    return sampler


# ====== 主训练入口 ======
def run_all_vulnerabilities(config_model,
                            vul_list,
                            base_path,
                            model_save_dir,
                            run_time,
                            epochs,
                            batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for vul in vul_list:
        # ["reentrancy", "delegatecall", "timestamp", "integeroverflow"]
        # if vul in ["reentrancy", "delegatecall", "integeroverflow"]:
        #     continue

        num_runs = run_time  # 重复次数
        metrics_list = []  # 保存每次测试指标
        for run_id in range(num_runs):
            print(f"\n--- Training on Vulnerability: {vul} | Run {run_id + 1}/{num_runs} ---")

            pyg_data_path = os.path.join(base_path, vul, "cfg_pyg_data")
            name_path = os.path.join(base_path, vul, f"final_{vul}_name.txt")
            label_path = os.path.join(base_path, vul, f"final_{vul}_label.txt")

            data_list = load_data(pyg_data_path, name_path, label_path)
            print(f"Loaded {len(data_list)} graphs.")

            # ====== 打印漏洞分布统计 ======
            count_stats(data_list, vul)

            # ====== 划分训练和测试集 ======
            # 提取标签列表
            labels = [int(data.y.item()) for data in data_list]

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

            train_loader = DataLoader(train_list, batch_size=batch_size, sampler=get_balanced_sampler(train_list))
            # train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_list, batch_size=batch_size)
            model = GraphModel(**config_model).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            train_losses, test_losses = [], []
            train_f1s, test_f1s = [], []
            train_precisions, test_precisions = [], []
            train_recalls, test_recalls = [], []

            best_f1 = 0
            patience = 15
            counter = 0

            for epoch in range(1, epochs + 1):
                train_loss, train_acc, train_prec, train_rec, train_f1 = train(model, train_loader, criterion,
                                                                               optimizer,
                                                                               device)
                print(
                    f"Epoch {epoch:02d} Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | P: {train_prec:.4f} | R: {train_rec:.4f} | "
                    f"F1: {train_f1:.4f}")
                test_loss, test_acc, test_prec, test_rec, test_f1 = test(model, test_loader, criterion, device)
                print(
                    f"Epoch {epoch:02d} Test  | Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | P: {test_prec:.4f} | R: {test_rec:.4f} | F1: {test_f1:.4f}")

                train_losses.append(train_loss)
                test_losses.append(test_loss)
                train_f1s.append(train_f1)
                test_f1s.append(test_f1)
                train_precisions.append(train_prec)
                test_precisions.append(test_prec)
                train_recalls.append(train_rec)
                test_recalls.append(test_rec)

                if test_f1 > best_f1:
                    best_f1 = test_f1
                    torch.save(model.state_dict(), f"../model_save/{model_save_dir}/{vul}_best_model.pth")
                    counter = 0
                else:
                    counter += 1
                    # if counter >= patience:
                    #     print("Early stopping triggered.")
                    #     break

            print(f"Finished training for {vul}.\n")
            plot_training_curves(
                train_losses, test_losses,
                train_f1s, test_f1s,
                train_precisions, test_precisions,
                train_recalls, test_recalls,
                vul=vul,
                save_path=f'../plots/{model_save_dir}/{vul}_{model_save_dir}_curve.png'
            )

            print(f"Start testing for {vul} Run {run_id + 1}.\n")
            model.load_state_dict(torch.load(f"../model_save/{model_save_dir}/{vul}_best_model.pth"))
            test_loss, test_acc, test_prec, test_rec, test_f1 = test(model, test_loader, criterion, device)
            metrics_list.append({
                'loss': test_loss,
                'acc': test_acc,
                'prec': test_prec,
                'rec': test_rec,
                'f1': test_f1
            })

        # 计算平均值
        avg_metrics = {
            'loss': np.mean([m['loss'] for m in metrics_list]),
            'acc': np.mean([m['acc'] for m in metrics_list]),
            'prec': np.mean([m['prec'] for m in metrics_list]),
            'rec': np.mean([m['rec'] for m in metrics_list]),
            'f1': np.mean([m['f1'] for m in metrics_list]),
        }

        # 输出平均结果
        result_str = (f"{vul} Avg Test | Loss: {avg_metrics['loss']:.4f} | Acc: {avg_metrics['acc']:.4f} | "
                      f"P: {avg_metrics['prec']:.4f} | R: {avg_metrics['rec']:.4f} | F1: {avg_metrics['f1']:.4f}\n")

        # # 找出 F1 最大的那组指标
        # best_metric = max(metrics_list, key=lambda m: m['f1'])
        #
        # # 构造输出字符串
        # result_str = (f"{vul} Best Test | Loss: {best_metric['loss']:.4f} | Acc: {best_metric['acc']:.4f} | "
        #               f"P: {best_metric['prec']:.4f} | R: {best_metric['rec']:.4f} | F1: {best_metric['f1']:.4f}\n")

        print(result_str)

        # 写入文件
        with open(f"../model_save/{model_save_dir}/test_results.txt", "a", encoding="utf-8") as f:
            f.write(result_str)


if __name__ == "__main__":
    vulnerability_list = ["reentrancy", "delegatecall", "timestamp", "integeroverflow"]

    config_model = {
        'vocab_size': 165,

        'hidden_dim_graph': 512,
        'net_type_graph': 'GGNN',
        'num_layers_graph': 2,
        'pool_type_graph': 'max',
        'dropout_rate_graph': 0.4
    }

    config_run = {
        'vul_list': ["reentrancy", "delegatecall", "timestamp", "integeroverflow"],
        'base_path': '../dataset',

        'model_save_dir': 'only_graph',
        'run_time': 1,
        'epochs': 100,
        'batch_size': 16
    }

    # 确保目录存在
    os.makedirs(f"../model_save/{config_run['model_save_dir']}", exist_ok=True)

    # 写日志文件（追加模式）
    log_path = f"../model_save/{config_run['model_save_dir']}/test_results.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("==== Config ====\n")
        for key, value in {**config_run, **config_model}.items():
            f.write(f"{key}: {value}\n")
        f.write("==== Results ====\n")

    run_all_vulnerabilities(config_model, **config_run)

