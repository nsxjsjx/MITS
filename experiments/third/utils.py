import os
import torch
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience=10):
        self.best_f1 = -1
        self.best_epoch = -1
        self.best_metrics = None
        self.counter = 0
        self.patience = patience
        self.stop = False

    def step(self, metrics, epoch):
        f1 = metrics["F1"]
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_epoch = epoch
            self.best_metrics = metrics.copy()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def load_dataset(vul):
    base = f"dataset/{vul}"
    names = open(f"{base}/final_{vul}_name.txt").read().splitlines()
    labels = list(map(int, open(f"{base}/final_{vul}_label.txt").read().splitlines()))

    data_map = {}
    for name, label in zip(names, labels):
        pt_name = name.replace(".sol", ".pt")
        data = torch.load(f"{base}/cfg_pyg_data_third/{pt_name}", weights_only=False)
        data.y = torch.tensor(label)
        data_map.setdefault(label, []).append(data)

    return data_map


def split_dataset(data_map, ratio=0.8):
    train, test = [], []

    for label, data_list in data_map.items():
        random.shuffle(data_list)
        k = int(len(data_list) * ratio)
        train += data_list[:k]
        test += data_list[k:]

    random.shuffle(train)
    random.shuffle(test)
    return train, test

class MetricRecorder:
    def __init__(self):
        self.train_loss = []
        self.acc = []
        self.pre = []
        self.rec = []
        self.f1 = []

    def update(self, loss, metrics):
        self.train_loss.append(loss)
        self.acc.append(metrics["ACC"])
        self.pre.append(metrics["PRE"])
        self.rec.append(metrics["REC"])
        self.f1.append(metrics["F1"])


def plot_curves(recorder, save_dir, title):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(recorder.train_loss) + 1)

    # ===== 1. Train Loss =====
    plt.figure()
    plt.plot(epochs, recorder.train_loss, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title(f"{title} - Train Loss")
    plt.grid(True)

    plt.savefig(os.path.join(save_dir, "loss.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, "loss.pdf"))
    plt.close()

    # ===== 2. Test Metrics =====
    plt.figure()
    plt.plot(epochs, recorder.acc, label="ACC", linewidth=2)
    plt.plot(epochs, recorder.pre, label="PRE", linewidth=2)
    plt.plot(epochs, recorder.rec, label="REC", linewidth=2)
    plt.plot(epochs, recorder.f1, label="F1", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"{title} - Test Metrics")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(save_dir, "metrics.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, "metrics.pdf"))
    plt.close()


def evaluate(y_true, y_pred):
    return {
        "ACC": accuracy_score(y_true, y_pred),
        "PRE": precision_score(y_true, y_pred),
        "REC": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
    }
