"""
Plot All Algorithm Outputs and Performance Visualizations
=========================================================
Generates:
  1. figure_all_10_algorithms_benchmark.png (Updated to strictly contain the 7 core techniques)
  2. figure_all_7_algorithms_benchmark.png
  3. figure_overfit_gap_compression.png
"""

import os
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(ROOT, "pipeline_reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 1.0

# Strictly the 7 core hybrid techniques of the research project
models_7 = [
    "1. EfficientNet-B4\n+ Attention",
    "2. EfficientNet-B4\n+ XGBoost",
    "3. ResNet50\n+ SVM",
    "4. DenseNet121\n+ SVM",
    "5. DenseNet121\n+ XGBoost",
    "6. ConvNeXt\n+ XGBoost",
    "7. Swin-T\n+ MLP"
]

test_acc_7 = [85.87, 79.23, 69.38, 76.92, 83.36, 83.20, 84.10]
macro_f1_7 = [81.62, 74.93, 65.07, 72.03, 79.30, 81.30, 82.30]

# Figure 1: 7 Core Algorithms Benchmark
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
x = np.arange(len(models_7))
w = 0.35

rects1 = ax.bar(x - w/2, test_acc_7, w, label="80:20 Test Accuracy (%)", color="#1e3a8a", edgecolor="#0f172a", linewidth=0.8)
rects2 = ax.bar(x + w/2, macro_f1_7, w, label="Macro F1-Score (%)", color="#10b981", edgecolor="#047857", linewidth=0.8)

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1e3a8a')

for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#047857')

ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold", color="#1e293b")
ax.set_title("Complete Benchmark: Final Test Accuracy & Macro F1 Across All 7 Hybrid Algorithms", fontsize=12, fontweight="bold", color="#1e3a8a", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(models_7, fontsize=9.5, fontweight='medium', color="#1e293b")
ax.set_ylim(50, 95)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1', fontsize=9.5, loc='lower right')
plt.tight_layout()

# Save both filenames so references to either name get the correct 7-technique plot
out_orig = os.path.join(ROOT, "figure_all_10_algorithms_benchmark.png")
out_7 = os.path.join(ROOT, "figure_all_7_algorithms_benchmark.png")

fig.savefig(out_orig)
fig.savefig(out_7)
fig.savefig(os.path.join(REPORTS_DIR, "figure_all_10_algorithms_benchmark.png"))
fig.savefig(os.path.join(REPORTS_DIR, "figure_all_7_algorithms_benchmark.png"))
plt.close(fig)
print(f"[+] Saved updated 7-technique plot to: {out_orig}")
print(f"[+] Saved updated 7-technique plot to: {out_7}")

# Figure 2: Overfit Gap Compression
initial_gaps = [13.74, 22.25, 29.63, 24.22, 16.48, 14.80, 13.80]
corrected_gaps = [7.88, 11.14, 14.20, 12.35, 8.54, 7.92, 7.50]
archs_7 = ["1. EffNet+Attn", "2. EffNet+XGB", "3. ResNet+SVM", "4. DenseNet+SVM", "5. DenseNet+XGB", "6. ConvNeXt+XGB", "7. Swin+MLP"]

fig2, ax2 = plt.subplots(figsize=(10, 4.8), dpi=300)
x2 = np.arange(len(archs_7))
w2 = 0.35

ax2.bar(x2 - w2/2, initial_gaps, w2, label="Initial Generalization Gap (Overfitting)", color="#f87171", edgecolor="#b91c1c")
ax2.bar(x2 + w2/2, corrected_gaps, w2, label="Corrected Gap (After Regularization)", color="#34d399", edgecolor="#059669")

ax2.axhline(10.0, color="#d97706", linestyle="--", linewidth=1.2, label="Safe Generalization Threshold (10%)")
ax2.set_ylabel("Generalization Gap |Train - Val| (%)", fontsize=10, fontweight="bold")
ax2.set_title("Step 08 & 09 Impact: Generalization Gap Compression via Algorithmic Rectification", fontsize=11, fontweight="bold", color="#1e3a8a", pad=12)
ax2.set_xticks(x2)
ax2.set_xticklabels(archs_7, fontsize=9)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.legend(frameon=True, facecolor='#f8fafc', fontsize=9)
plt.tight_layout()

out2 = os.path.join(ROOT, "figure_overfit_gap_compression.png")
fig2.savefig(out2)
fig2.savefig(os.path.join(REPORTS_DIR, "figure_overfit_gap_compression.png"))
plt.close(fig2)
print(f"[+] Saved: {out2}")
