"""Metrics, tables and plots."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)


def metrics(y, p):
    return dict(acc=100 * accuracy_score(y, p),
                precision=100 * precision_score(y, p, average="macro", zero_division=0),
                recall=100 * recall_score(y, p, average="macro", zero_division=0),
                f1=100 * f1_score(y, p, average="macro", zero_division=0))


def show_table(df, title, out_dir, fname):
    line = "=" * 130
    print(f"\n{line}\n{title}\n{line}")
    print(df.to_string(index=False))
    print(line)
    df.to_csv(os.path.join(out_dir, fname + ".csv"), index=False)

    # Wrap long cells so table text stays inside its column in the PNG.
    import textwrap
    widths = {
        "Stage": 18,
        "Correction": 27,
        "Diagnosis": 15,
    }
    display = []
    max_lines = 1
    for row in df.itertuples(index=False, name=None):
        wrapped = []
        for col, value in zip(df.columns, row):
            value = str(value)
            # Keep the PNG readable; the CSV retains the full correction detail.
            if col == "Correction" and "rejected:" in value:
                value = "Regularization rejected; initial model kept"
            cell = textwrap.fill(value, width=widths.get(col, 12), break_long_words=True)
            wrapped.append(cell)
            max_lines = max(max_lines, cell.count("\n") + 1)
        display.append(wrapped)

    fig, ax = plt.subplots(figsize=(max(15, 1.7 * len(df.columns)), max(4.5, 0.85 * (len(df) + 3))))
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=16)
    col_widths = []
    for col in df.columns:
        if col == "Correction":
            col_widths.append(0.23)
        elif col == "Stage":
            col_widths.append(0.14)
        elif col == "Diagnosis":
            col_widths.append(0.13)
        else:
            col_widths.append(0.075)
    total = sum(col_widths)
    col_widths = [w / total for w in col_widths]
    tb = ax.table(
        cellText=display,
        colLabels=list(df.columns),
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
    )
    tb.auto_set_font_size(False)
    tb.set_fontsize(7.5)
    tb.scale(1, max(2.2, min(3.5, 0.7 * max_lines)))
    for (r, c), cell in tb.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", fontweight="bold")
    fig.savefig(os.path.join(out_dir, fname + ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

def plot_curve(curve, path, title):
    if not curve:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for k in ("train_loss", "val_loss"):
        if k in curve:
            ax[0].plot(curve[k], label=k)
    ax[0].set_title(title + " - loss")
    ax[0].legend()
    for k in ("train_acc", "val_acc"):
        if k in curve:
            ax[1].plot(curve[k], label=k)
    ax[1].set_title("accuracy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def confusion_and_report(y, p, classes, out_dir, title):
    cm = confusion_matrix(y, p, labels=range(len(classes)))
    n = len(classes)
    s = min(max(6, n * 0.25), 22)
    fig, ax = plt.subplots(figsize=(s, s))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if n <= 25:
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(classes, rotation=90, fontsize=7); ax.set_yticklabels(classes, fontsize=7)
        for i in range(n):
            for j in range(n):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix_test.png"), dpi=130)
    plt.close(fig)
    rep = classification_report(y, p, labels=range(n), target_names=classes, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(os.path.join(out_dir, "per_class_report_test.csv"))
