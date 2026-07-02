import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Data
# ===============================
variants = [
    "w/o GCN",
    "w/o GAT",
    "w/o GGNN",
    "w/o GraphSAGE",
    "w/o SF",
    "Ours"
]
vulnerabilities = ['Reentrancy', 'Timestamp', 'Arithmetic', 'Delegatecall']

f1 = np.array([
    [77.71, 86.99, 78.87, 88.69],
    [81.00, 86.02, 79.76, 91.52],
    [80.03, 87.47, 80.29, 89.38],
    [73.02, 85.61, 78.88, 91.08],
    [77.30, 87.44, 78.86, 88.77],
    [81.19, 87.76, 80.30, 92.71]
])

# ===============================
# Color palette (soft & paper-friendly)
# ===============================
colors = [
    '#E64B35',  # red
    '#4DBBD5',  # blue
    '#00A087',  # green
    '#3C5488',  # navy
    '#FFB04D',  # orange — 新增！明亮但不刺眼，与现有色区分度高
    '#F39B7F',  # Ours (highlight, peach/salmon)
]

plt.rcParams.update({
    'font.size': 13,
    # 'axes.labelsize': 9,
    # 'axes.titlesize': 13,
    'legend.fontsize': 11,
    'pdf.fonttype': 42,
})

# ===============================
# Grouped Bar Chart (F1)
# ===============================
x = np.arange(len(vulnerabilities))
width = 0.15

fig, ax = plt.subplots(figsize=(9, 4))

for i, variant in enumerate(variants):
    ax.bar(
        x + (i - 2) * width,
        f1[i],
        width=width,
        label=variant,
        color=colors[i],
        edgecolor='black' if variant == 'Ours' else 'none',
        linewidth=1.0 if variant == 'Ours' else 0
    )

ax.set_ylabel('F1-score (%)')
ax.set_xlabel('Vulnerability Type')
ax.set_xticks(x)
ax.set_xticklabels(vulnerabilities)
ax.set_ylim(70, 100)
ax.set_yticks([70, 80, 90, 100])
ax.legend(ncol=6, loc='upper center', bbox_to_anchor=(0.5, 1.15))
ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('ablation_f1.pdf', format='pdf', dpi=300)
plt.show()

# ===============================
# Delta F1 Bar Chart (Ours - Ablation)
# ===============================
delta = f1[-1] - f1[:-1]  # Ours minus ablations

fig, ax = plt.subplots(figsize=(9, 4))

bar_width = 0.18
x = np.arange(len(vulnerabilities))

for i, variant in enumerate(variants[:-1]):
    ax.bar(
        x + (i - 1.5) * bar_width,
        delta[i],
        width=bar_width,
        label=variant,
        color=colors[i]
    )

ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('ΔF1 (Ours − Ablation)')
ax.set_xlabel('Vulnerability Type')
ax.set_xticks(x)
ax.set_xticklabels(vulnerabilities)
ax.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.15))
ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('ablation_f1_delta.pdf', format='pdf', dpi=300)
plt.show()
