import torch
import random
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from sklearn.utils.class_weight import compute_class_weight

# todo
from model.third.ablation.wogat import MultiGNN_CFG_Model
from experiments.third.utils import load_dataset, split_dataset, evaluate, EarlyStopping, MetricRecorder, plot_curves

device = torch.device("cpu")  # Mac 上强烈建议 CPU


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, optimizer, class_weights):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, _ = model(data)
        loss = torch.nn.functional.cross_entropy(out, data.y, weight=class_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# @torch.no_grad()
# def test(model, loader):
#     model.eval()
#     y_true, y_pred = [], []
#     for data in loader:
#         data = data.to(device)
#         out, _ = model(data)
#         pred = out.argmax(dim=1)
#         y_true.append(data.y.item())
#         y_pred.append(pred.item())
#     return evaluate(y_true, y_pred)
@torch.no_grad()
def test(model, loader):
    model.eval()
    y_true, y_pred = [], []
    for data in loader:
        data = data.to(device)
        out, _ = model(data)
        pred = out.argmax(dim=1)

        # 修改点：使用 extend 配合 tolist() 处理整个 batch
        # 如果 data.y 是标量（batch_size=1），tolist() 也会正常工作
        y_true.extend(data.y.view(-1).cpu().tolist())
        y_pred.extend(pred.view(-1).cpu().tolist())

    return evaluate(y_true, y_pred)


batch_size = 16
hidden_dim = 128
struct_dim = 4
lr = 1e-3
# todo
patience = 10
epoch_num = 200
vocab_size = 165
token_dim = 128


def run(vul, seed):
    print(f"\n===== Running {vul} | seed={seed} =====")

    data_map = load_dataset(vul)
    train_set, test_set = split_dataset(data_map)

    print(f"Train size: {len(train_set)}, Test size: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = MultiGNN_CFG_Model(
        vocab_size=vocab_size,
        token_dim=token_dim,
        hidden_dim=hidden_dim,
        struct_dim=struct_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    recorder = MetricRecorder()
    stopper = EarlyStopping(patience=patience)

    y_train = [data.y.item() for data in train_set]
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    for epoch in range(1, epoch_num + 1):
        loss = train_one_epoch(model, train_loader, optimizer, class_weights)
        metrics = test(model, test_loader)

        recorder.update(loss, metrics)

        print(
            f"[{epoch:03d}] "
            f"Loss={loss:.4f} "
            f"ACC={metrics['ACC']:.4f} "
            f"PRE={metrics['PRE']:.4f} "
            f"REC={metrics['REC']:.4f} "
            f"F1={metrics['F1']:.4f}"
        )

        stopper.step(metrics, epoch)
        if stopper.stop:
            print(
                f"Early stopping at epoch {epoch}, "
                f"best F1={stopper.best_f1:.4f} "
                f"(epoch {stopper.best_epoch})"
            )
            break

    # todo
    save_dir = f"results/third/{vul}/ablation/wogat/seed_{seed}"
    plot_curves(recorder, save_dir, title=f"{vul} (seed={seed})")

    return stopper.best_metrics


import os
import numpy as np

if __name__ == "__main__":
    for vul in ["delegatecall", "integeroverflow", "timestamp", "reentrancy"]:
        print(f"\n================ {vul.upper()} ================")

        all_results = {
            "ACC": [],
            "PRE": [],
            "REC": [],
            "F1": []
        }

        for seed in range(5):
            print(f"\n######## Seed {seed} ########")
            set_seed(seed)
            best_metrics = run(vul, seed)

            for k in all_results:
                all_results[k].append(best_metrics[k])

        # 计算 mean ± std
        summary = {}
        for k, values in all_results.items():
            mean = np.mean(values)
            std = np.std(values)
            summary[k] = (mean, std)

        # 打印
        print(f"\n>>> Final Results for {vul}")
        for k, (m, s) in summary.items():
            print(f"{k}: {m:.4f} ± {s:.4f}")

        # 保存
        # todo
        save_dir = f"results/third/{vul}/ablation/wogat"
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, "summary.txt"), "w") as f:
            for k, (m, s) in summary.items():
                f.write(f"{k}: {m:.4f} ± {s:.4f}\n")

