import matplotlib.pyplot as plt
import numpy as np
import os

# Data
metrics = ['AP', 'Precision', 'Recall', 'F1-Score']
thresholds = ['0.5', '0.75', '0.9']

baseline_scores = {
    '0.5': [0.845, 0.957, 0.868, 0.910],
    '0.75': [0.667, 0.821, 0.745, 0.781],
    '0.9': [0.379, 0.537, 0.487, 0.511]
}

twostream_scores = {
    '0.5': [0.851, 0.960, 0.875, 0.915],
    '0.75': [0.665, 0.820, 0.747, 0.782],
    '0.9': [0.340, 0.457, 0.416, 0.436]
}

x = np.arange(len(metrics))
width = 0.35

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

for i, thresh in enumerate(thresholds):
    ax = axes[i]
    rects1 = ax.bar(x - width/2, baseline_scores[thresh], width, label='Baseline (Original)', color='skyblue', edgecolor='black')
    rects2 = ax.bar(x + width/2, twostream_scores[thresh], width, label='Two-Encoder (Proposed)', color='salmon', edgecolor='black')
    
    ax.set_title(f'IoU @ {thresh}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.1)
    if i == 0:
        ax.set_ylabel('Score', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig('/workspace/TwoStream-GeneSegNet/Results/metrics_comparison.png', dpi=300)
print("Grafik kaydedildi: /workspace/TwoStream-GeneSegNet/Results/metrics_comparison.png")
