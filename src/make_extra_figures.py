"""Two additional publication figures for the MDPI Sensors submission:
   - pipeline / system overview schematic
   - 10-class ward-transition confusion matrix heatmap
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

mpl.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9})
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
C_STATIC, C_TRANS, C_BED, C_BLUE = "#4C72B0", "#DD8452", "#C44E52", "#4C72B0"


def pipeline():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

    def box(x, y, w, h, text, tc="black", fs=8.5, bold_first=False):
        # clean monochrome: white fill, dark border (g001 style)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                    fc="white", ec="#222", lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.3, color="#222"))

    # top row: data -> preprocessing -> representations -> evaluation
    box(1, 30, 17, 12, "Public datasets\nHAPT (30 subj, 6-axis IMU)\nMHEALTH (10 subj,\nIMU + chest ECG)\n50 Hz", fs=8)
    box(21, 30, 17, 12, "Windowing\n2.56 s, segment-bounded\nWard taxonomy:\n4 static + 6 directional\ntransitions", fs=8)
    box(41, 33, 26, 9, "Representations\n(same input, same protocol)", fs=8.5)
    box(42, 30.3, 12, 3.4, "Hand-crafted 3/6-axis\n+ XGBoost", fs=7)
    box(55, 30.3, 11, 3.4, "UniMTS FM\n(probe / fine-tune)", fs=7)
    box(70, 30, 28, 12, "Subject-grouped CV\nper-subject macro-F1\npaired Wilcoxon + CI\nlabel-efficiency curve", fs=8)
    for x1, x2 in [(18, 21), (38, 41), (67, 70)]:
        arrow(x1, 36, x2, 36)

    # bottom row: the four questions / outputs
    box(1, 6, 22, 16, "Q1  transition difficulty\nQ2  which sensor axes\nQ3  labelling cost\nQ4  does pretraining help", fs=8.5)
    box(28, 6, 30, 16, "Findings\n• bed-exit = weakest class\n• gyroscope carries the signal\n• event-bound labelling\n• 6-axis hand-crafted > FM", fs=8.5)
    box(63, 6, 35, 16, "Ward implication\nsensor choice (gyroscope) > model scale;\nECG quality falls with activity\n(motion-induced false-alarm mechanism)", fs=8.5)
    arrow(23, 14, 28, 14); arrow(58, 14, 63, 14)
    arrow(50, 30, 50, 22)  # analysis -> findings

    ax.text(50, 44.5, "Study pipeline and questions", ha="center", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig_pipeline.png", bbox_inches="tight")
    plt.close(fig)
    print("pipeline done")


def confusion():
    df = pd.read_csv(RES / "trans_confusion_w128.csv", index_col=0)
    labels = [c.replace("_", "-").title().replace("-To-", "-to-") for c in df.index]
    cm = df.values.astype(float)
    row_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    im = ax.imshow(row_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm[i, j]
            if v > 0:
                ax.text(j, i, int(v), ha="center", va="center", fontsize=6.5,
                        color="white" if row_norm[i, j] > 0.5 else "#333")
    # highlight the sitting<->standing cells
    for (i, j) in [(1, 2), (2, 1)]:
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=C_BED, lw=2.2))
    cb = fig.colorbar(im, fraction=0.046, pad=0.04)
    cb.set_label("row-normalised proportion")
    ax.set_title("Ward 10-class confusion (2.56 s, six-axis)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_confusion.png", bbox_inches="tight")
    plt.close(fig)
    print("confusion done")


if __name__ == "__main__":
    pipeline()
    confusion()
