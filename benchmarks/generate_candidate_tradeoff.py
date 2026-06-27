import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load existing data
metrics_path = "temp/benchmark_data/candidate_metrics.json"
if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        data = json.load(f)
    results = data["results"]
else:
    print("No candidate metrics data found, using default values")
    results = [
        {"k_candidate": 10, "workload_reduction": 0.9999, "recall": 0.6654676258992805},
        {"k_candidate": 50, "workload_reduction": 0.9995, "recall": 0.6931654676258993},
        {"k_candidate": 100, "workload_reduction": 0.999, "recall": 0.7780575539568346},
        {"k_candidate": 200, "workload_reduction": 0.998, "recall": 0.9028776978417267},
        {"k_candidate": 500, "workload_reduction": 0.995, "recall": 0.960431654676259},
        {"k_candidate": 1000, "workload_reduction": 0.99, "recall": 1.0},
        {"k_candidate": 2000, "workload_reduction": 0.98, "recall": 1.0},
        {"k_candidate": 5000, "workload_reduction": 0.95, "recall": 1.0},
    ]

# Plot trade-off Elbow Curve
plt.style.use('dark_background')
fig, ax1 = plt.subplots(figsize=(9, 5), facecolor='#0b0e14')
ax1.set_facecolor('#0b0e14')

x = [r["k_candidate"] for r in results]
y_recall = [r["recall"] * 100.0 for r in results]
y_reduction = [r["workload_reduction"] * 100.0 for r in results]

color_recall = '#00f2fe'  # Cyan
color_reduction = '#ff007f'  # Neon Pink

ax1.set_xlabel('Candidate Size (K)', color='#8b949e')
ax1.set_ylabel('Recall of Top-10 Pits (%)', color=color_recall)
ax1.plot(x, y_recall, color=color_recall, marker='o', linewidth=2.5, label='Recall of Top-10 Pits')
ax1.tick_params(axis='y', labelcolor=color_recall)
ax1.set_xscale('log')
ax1.set_xticks(x)
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax2 = ax1.twinx()
ax2.set_ylabel('Mask R-CNN Workload Reduction (%)', color=color_reduction)
ax2.plot(x, y_reduction, color=color_reduction, marker='s', linestyle='--', linewidth=2, label='Workload Reduction')
ax2.tick_params(axis='y', labelcolor=color_reduction)

plt.title('Downstream Workload Reduction vs. Target Recall Trade-Off', color='#c9d1d9', fontsize=12, pad=15)
plt.grid(True, color='#21262d', linestyle=':', alpha=0.6)

os.makedirs("assets", exist_ok=True)
plt.savefig("assets/candidate_tradeoff.png", dpi=150, bbox_inches='tight', facecolor='#0b0e14')
plt.savefig("assets/candidate_tradeoff.svg", bbox_inches='tight', facecolor='#0b0e14')
print("Trade-off plot saved to assets/candidate_tradeoff.png and assets/candidate_tradeoff.svg")
