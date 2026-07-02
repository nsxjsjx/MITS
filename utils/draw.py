import os

import matplotlib.pyplot as plt


def plot_loss_curve(train_losses, vul_type="unknown"):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Loss Curve for {vul_type}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'../loss_figures/loss_curve_{vul_type}.png')
    plt.show()

def plot_training_curves(
    train_losses, test_losses,
    train_f1s, test_f1s,
    train_precisions, test_precisions,
    train_recalls, test_recalls,
    vul=None,
    save_path=None
):
    """
    绘制训练和测试过程中的 Loss、Precision、Recall 和 F1 曲线

    参数：
    - train_losses, test_losses: 每轮 Loss 列表
    - train_f1s, test_f1s: 每轮 F1 列表
    - train_precisions, test_precisions: 每轮 Precision 列表
    - train_recalls, test_recalls: 每轮 Recall 列表
    - vul: str, 漏洞名（可选）
    - save_path: str, 图片保存路径（可选）
    """

    title_prefix = f"[{vul}] " if vul else ""
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(18, 10))

    # Loss 曲线
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs_range, test_losses, label='Test Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title_prefix}Loss Curve')
    plt.legend()
    plt.grid(True)

    # F1 曲线
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, train_f1s, label='Train F1', marker='o')
    plt.plot(epochs_range, test_f1s, label='Test F1', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title(f'{title_prefix}F1 Score Curve')
    plt.legend()
    plt.grid(True)

    # Precision 曲线
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, train_precisions, label='Train Precision', marker='o')
    plt.plot(epochs_range, test_precisions, label='Test Precision', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title(f'{title_prefix}Precision Curve')
    plt.legend()
    plt.grid(True)

    # Recall 曲线
    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, train_recalls, label='Train Recall', marker='o')
    plt.plot(epochs_range, test_recalls, label='Test Recall', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title(f'{title_prefix}Recall Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"图像已保存到：{save_path}")

    plt.show()


