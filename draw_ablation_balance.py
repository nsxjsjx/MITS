import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# =========================
# Aggregation strategies
# =========================
strategies = [
    "w/o GCN",
    "w/o GAT",
    "w/o GGNN",
    "w/o GraphSAGE",
    "w/o SF",
    "Ours"
]

# =========================
# PRE / RE data
# =========================
data = {
    "Reentrancy": {
        "PRE": [72.44, 82.53, 73.81, 71.60, 73.90, 84.38],
        "RE":  [85.33, 80.00, 88.00, 77.33, 81.33, 78.67],
    },
    "Timestamp": {
        "PRE": [86.18, 82.35, 86.32, 82.32, 84.08, 85.02],
        "RE":  [88.33, 91.11, 88.89, 89.44, 91.07, 91.11],
    },
    "Arithmetic": {
        "PRE": [76.05, 76.39, 77.51, 74.66, 76.57, 86.07],
        "RE":  [83.33, 84.44, 83.33, 84.44, 82.22, 76.67],
    },
    "Delegatecall": {
        "PRE": [88.48, 91.49, 94.43, 89.06, 86.03, 89.64],
        "RE":  [89.23, 92.31, 86.15, 93.85, 92.31, 96.92],
    }
}

# =========================
# Style per vulnerability
# =========================
styles = {
    "Reentrancy": {
        "marker": "o",
        "color": "#1f77b4"
    },
    "Timestamp": {
        "marker": "s",
        "color": "#ff7f0e"
    },
    "Arithmetic": {
        "marker": "^",
        "color": "#2ca02c"
    },
    "Delegatecall": {
        "marker": "D",
        "color": "#d62728"
    }
}

# =========================
# Generate one PDF per vulnerability
# =========================
for vuln, vals in data.items():

    pre = np.array(vals["PRE"])
    re = np.array(vals["RE"])

    # Adaptive axis range
    min_val = min(pre.min(), re.min())
    max_val = max(pre.max(), re.max())
    padding = (max_val - min_val) * 0.08

    x_min, x_max = min_val - padding, max_val + padding
    y_min, y_max = min_val - padding, max_val + padding

    output_path = f"results/third/ablation_{vuln}_balance.pdf"

    with PdfPages(output_path) as pdf:
        plt.figure(figsize=(7, 7))

        for i, strategy in enumerate(strategies):
            is_ours = (strategy == "Ours")
            # is_mean = (strategy == "Mean Pooling")
            # is_median = (strategy == "Median Pooling")

            plt.scatter(
                pre[i],
                re[i],
                marker=styles[vuln]["marker"],
                color=styles[vuln]["color"],
                s=660 if is_ours else 560,
                edgecolors="black" if is_ours else "none",
                linewidths=1.5 if is_ours else 0.6,
                alpha=0.95 if is_ours else 0.8,
                zorder=6 if is_ours else 4
            )

            # if is_mean:
            #     offset = (-40, 10)
            # elif is_median:
            #     offset = (3, 10)
            # else:
            #     offset = (6, 6)
            offset = (6, 6)

            plt.annotate(
                strategy,
                (pre[i], re[i]),
                fontsize=15 if is_ours else 13,
                xytext=offset,
                textcoords="offset points"
            )

        # PRE = RE reference line
        ref_min = min(x_min, y_min)
        ref_max = max(x_max, y_max)
        plt.plot(
            [ref_min, ref_max],
            [ref_min, ref_max],
            linestyle="--",
            linewidth=1.2,
            color="gray",
            alpha=0.6,
            label="PRE = RE"
        )

        plt.xlabel("Precision")
        plt.ylabel("Recall")
        plt.title(
            "",
            fontsize=13
        )

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(loc="lower right")

        pdf.savefig()
        plt.close()

    print(f"Saved: {output_path}")
