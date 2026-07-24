"""Two additional publication figures for the MDPI Sensors submission:
   - pipeline / system overview schematic (top-to-bottom flowchart)
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

NL = chr(10)
ARROW = chr(8594)          # right arrow

ACCENT = "#1F4E79"         # stage badges, spine, closing band
SOFT = "#EEF3F9"           # stage fill
EDGE = "#9DB2C8"


def pipeline(compact=False, outname="fig_pipeline.png"):
    """Top-to-bottom flow of the study, from dataset to ward implication.

    Monochrome by design: the flow is carried by position and arrows, so no
    stage numbers or colour are needed and the figure prints cleanly in
    greyscale. compact=True tightens the vertical gaps.
    """
    if compact:
        # gaps of 6 between stages so the connecting arrows are clearly visible;
        # the boxes are correspondingly shorter, keeping the figure compact.
        # uniform 6-unit gaps between stages; stage 3 is taller because it also
        # carries the bed-exit call-out.
        # the evaluation box carries four lines (it also states the labelling
        # requirement analysis), so it is taller than the others.
        L = dict(s1=(111.5, 16), s2=(89.5, 16), s3=(60.5, 23), s45=(38.5, 16),
                 s5=(13, 19.5), label=57.5, badge=64.5, ymax=129.5, fh=9.9)
    else:
        L = dict(s1=(118, 16), s2=(96, 16), s3=(64, 23), s45=(41, 16),
                 s5=(13, 19.5), label=60.5, badge=68.0, ymax=136, fh=10.4)

    fig, ax = plt.subplots(figsize=(7.6, L["fh"]))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, L["ymax"])
    ax.axis("off")

    # single-column stages span X_FULL; the parallel pair splits the same span
    X0, W_FULL = 6, 88
    W_HALF = 42
    X_RIGHT = X0 + W_FULL - W_HALF

    INK, LINE, FILL = "#111111", "#333333", "#FFFFFF"

    def stage(y, h, title, lines, x=X0, w=W_FULL):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.6,rounding_size=1.6",
                                    fc=FILL, ec=LINE, lw=1.3, zorder=2))
        ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="top",
                fontsize=10.5, fontweight="bold", color=INK, zorder=3)
        ax.text(x + w / 2, y + h - 8.4, NL.join(lines), ha="center", va="top",
                fontsize=8.3, color=INK, linespacing=1.55, zorder=3)

    def flow(y_from, y_to, x=50):
        # start/end just outside the rounded-box padding so the shaft is visible
        ax.add_patch(FancyArrowPatch((x, y_from - 0.8), (x, y_to + 0.8),
                                     arrowstyle="-|>", mutation_scale=15, lw=1.6,
                                     color=LINE, zorder=1))

    # dataset
    stage(*L["s1"], "Public dataset  -  HAPT", [
        "30 volunteers (19-48 y) - waist-worn smartphone",
        "triaxial accelerometer + triaxial gyroscope, 50 Hz",
        "rare public corpus: transition direction AND gyroscope"])
    flow(L["s1"][0], L["s2"][0] + L["s2"][1])

    # windowing
    stage(*L["s2"], "Windowing", [
        "2.56 s windows (128 samples), 50% overlap",
        "placed strictly inside one labelled activity bout",
        "additionally swept 0.64-2.56 s, survivorship-controlled"])
    flow(L["s2"][0], L["s3"][0] + L["s3"][1])

    # ward task, with the bed-exit definition called out inside the box
    stage(*L["s3"], "Ten-class ward task", [
        "four basic activities: lying, sitting, standing, walking",
        "six directional postural transitions",
        "stairs excluded - absent in a ward, inflate accuracy"])
    ax.text(50, L["badge"],
            "bed-exit  =  lie" + ARROW + "sit     lie" + ARROW + "stand",
            ha="center", va="center", fontsize=8.6, fontweight="bold",
            color=INK, zorder=4,
            bbox=dict(boxstyle="round,pad=0.4", fc=FILL, ec=INK, lw=1.4))
    flow(L["s3"][0], L["s45"][0] + L["s45"][1])

    # the two representations run in parallel, not in sequence
    # set beside the arrow so it does not cover it; the wording matches the
    # figure caption. "same" distributes over all three nouns.
    ax.text(53, L["label"], "same windows, folds and classifier settings",
            ha="left", va="center", fontsize=7.6, style="italic",
            color=LINE, zorder=3)
    stage(*L["s45"], "Hand-crafted features", [
        "150 features (6-axis) / 75 (3-axis)",
        "gravity retained " + ARROW + " encodes posture",
        "XGBoost, fixed untuned settings"], x=X0, w=W_HALF)
    stage(*L["s45"], "UniMTS foundation model", [
        "frozen probe / embeddings + XGBoost",
        "full fine-tuning",
        "audit: 3 input channels = accel only"], x=X_RIGHT, w=W_HALF)
    y_join = L["s5"][0] + L["s5"][1] + 1.0
    ax.add_patch(FancyArrowPatch((X0 + W_HALF / 2, L["s45"][0] - 0.8), (46, y_join),
                                 arrowstyle="-|>", mutation_scale=13, lw=1.4,
                                 color=LINE, zorder=1))
    ax.add_patch(FancyArrowPatch((X_RIGHT + W_HALF / 2, L["s45"][0] - 0.8), (54, y_join),
                                 arrowstyle="-|>", mutation_scale=13, lw=1.4,
                                 color=LINE, zorder=1))

    # evaluation
    stage(*L["s5"], "Evaluation", [
        "subject-grouped 5-fold cross-validation",
        "per-subject macro-F1 and bed-exit F1 (n = 30)",
        "paired Wilcoxon, bootstrap CI, Holm correction",
        "labelling requirement swept from 5 min to the full budget"])
    flow(L["s5"][0], 7.2)

    # closing statement, inverted for emphasis but still greyscale
    ax.add_patch(FancyBboxPatch((X0, 0), W_FULL, 7,
                                boxstyle="round,pad=0.6,rounding_size=1.6",
                                fc=INK, ec="none", zorder=2))
    ax.text(50, 3.5,
            "Ward implication: sensor choice (gyroscope) over model scale;" + NL +
            "size the labelling budget by target bed-exit event count",
            ha="center", va="center", fontsize=8.8, fontweight="bold",
            color="white", linespacing=1.5, zorder=3)

    fig.savefig(FIG / outname, bbox_inches="tight")
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
    # The compact layout is the one used as Figure 1 in the manuscript; it keeps
    # the same stages and box heights but tightens the gaps so the figure fits a
    # manuscript page. pipeline(compact=False) renders the taller variant.
    pipeline(compact=True)
    confusion()
