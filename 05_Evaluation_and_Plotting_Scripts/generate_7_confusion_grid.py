"""
Generate Consolidated 7-Technique Domain Confusion Matrix Grid Figure
====================================================================
Outputs: figure_all_7_confusion_matrices_domain_grid.png
Combines the 3x3 domain botanical confusion matrices (Medicinal, Endangered, Toxic)
for all 7 hybrid deep learning techniques into one publication-ready figure.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(ROOT, "pipeline_reports")

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

techniques = [
    {"folder": "01_EfficientNetB4_Attention", "name": "1. EfficientNet-B4 + Attention", "acc": "88.14%"},
    {"folder": "02_EfficientNetB4_XGBoost",   "name": "2. EfficientNet-B4 + XGBoost",   "acc": "80.79%"},
    {"folder": "03_ResNet50_SVM",             "name": "3. ResNet50 + SVM",             "acc": "66.12%"},
    {"folder": "04_DenseNet121_SVM",          "name": "4. DenseNet121 + SVM",          "acc": "75.60%"},
    {"folder": "05_DenseNet121_XGBoost",      "name": "5. DenseNet121 + XGBoost",      "acc": "83.85%"},
    {"folder": "06_ConvNeXt_XGBoost",         "name": "6. ConvNeXt + XGBoost",         "acc": "83.90%"},
    {"folder": "07_SwinTransformer_MLP",     "name": "7. Swin Transformer + MLP",     "acc": "84.65%"}
]

# Create a 2x4 grid (7 plots + 1 summary text panel)
fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=300)
axes = axes.flatten()

for idx, tech in enumerate(techniques):
    ax = axes[idx]
    img_path = os.path.join(ROOT, "plant_hybrid", tech["folder"], "outputs", "5_fold", "confusion_matrix_domain_3x3.png")
    if os.path.exists(img_path):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"{tech['name']} (Acc: {tech['acc']})", fontsize=11, fontweight='bold', pad=6, color="#1e3a8a")
    else:
        ax.text(0.5, 0.5, f"Image not found:\n{tech['folder']}", ha='center', va='center', fontsize=9)
        ax.axis('off')

# 8th panel: Summary benchmark table
ax_summary = axes[7]
ax_summary.axis('off')

summary_text = (
    "Cross-Architecture Summary\n"
    "===========================\n"
    "• Evaluated Classes: 111 Species\n"
    "• Primary Domains: 3 Supercategories\n"
    "  - Medicinal Plants\n"
    "  - Endangered Flora (CSFID)\n"
    "  - Toxic Chinese Herbs\n\n"
    "Performance Ranking:\n"
    "  #1 EfficientNet-B4 + Attn : 88.14%\n"
    "  #2 Swin-T + MLP          : 84.65%\n"
    "  #3 ConvNeXt + XGBoost    : 83.90%\n"
    "  #4 DenseNet121 + XGBoost : 83.85%\n"
    "  #5 EfficientNet-B4 + XGB : 80.79%\n"
    "  #6 DenseNet121 + SVM     : 75.60%\n"
    "  #7 ResNet50 + SVM        : 66.12%\n\n"
    "Protocol: Stratified 5-Fold CV"
)

ax_summary.text(
    0.05, 0.5, summary_text,
    fontsize=9.5, fontfamily='monospace', verticalalignment='center',
    bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.2)
)

fig.suptitle(
    "Cross-Technique Domain Confusion Matrix Comparison Across All 7 Hybrid Architectures (5-Fold Cross-Validation)",
    fontsize=14, fontweight='bold', y=0.98, color="#0f172a"
)
fig.tight_layout(rect=[0, 0, 1, 0.96])

out_path = os.path.join(ROOT, "figure_all_7_confusion_matrices_domain_grid.png")
fig.savefig(out_path, bbox_inches="tight")
fig.savefig(os.path.join(REPORTS_DIR, "figure_all_7_confusion_matrices_domain_grid.png"), bbox_inches="tight")
plt.close(fig)
print(f"[+] Successfully generated consolidated grid figure: {out_path}")

