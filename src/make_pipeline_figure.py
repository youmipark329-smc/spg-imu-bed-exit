"""Figure 1: study pipeline as a plain vector block diagram (revision 4: monochrome, restrained).

Round-3 editorial comment: the figure should use standard editable vector
elements and a simplified design with consistent fonts, line widths, arrows,
icons, spacing and colour. This version therefore uses one typeface (Arial),
one line width (0.75 pt) for every box, connector and arrow, one arrowhead
style, one outline colour, no icons, no fills, and uniform spacing. Layout
coordinates are in inches at the 6.5 in placement width.
Outputs (figures/): fig_pipeline.svg (editable vector, text kept as text),
fig_pipeline.pdf, fig_pipeline.eps, fig_pipeline.png and .tif (600 dpi).
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

mpl.rcParams.update({
    "savefig.dpi": 600, "font.family": "Arial",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"

INK = "#000000"          # single colour: black
TEXT = "#000000"
LW = 0.5                 # single line width (pt) for boxes, connectors, arrows
TITLE, BODY, HEAD = 8, 8, 8
LINE = 0.14              # baseline spacing for 8 pt text (in)
W, H = 6.5, 4.4


def box(ax, x, y, w, h, title, lines):
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=INK, lw=LW))
    n = 1 + len(lines)
    top = y + h / 2 + (n - 1) * LINE / 2
    ax.text(x + w / 2, top, title, ha="center", va="center", fontsize=TITLE, fontweight="bold", color=TEXT)
    for i, ln in enumerate(lines, 1):
        ax.text(x + w / 2, top - i * LINE, ln, ha="center", va="center", fontsize=BODY, color=TEXT)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=LW, color=INK,
                                 shrinkA=0, shrinkB=0, joinstyle="miter"))


def line(ax, p, q):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=INK, lw=LW, solid_capstyle="projecting")


def main():
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")

    # ---- column geometry (inches); equal 0.45 in channels between columns for the connectors
    LX, LW_ = 0.05, 1.95
    MX, MW = 2.35, 2.05
    RX, RW = 4.70, 1.80
    HEAD_Y = 4.25
    for x, w, t in ((LX, LW_, "Stage 1: data and task"), (MX, MW, "Stage 2: representations"),
                    (RX, RW, "Stages 3 and 4: scoring")):
        ax.text(x + w / 2, HEAD_Y, t, ha="center", va="center", fontsize=HEAD, color=TEXT)

    # ---- left column: three boxes, equal height and equal gap
    lh, gap = 0.95, 0.30
    lys = [3.05, 3.05 - lh - gap, 3.05 - 2 * (lh + gap)]
    box(ax, LX, lys[0], LW_, lh, "HAPT dataset", ["30 subjects", "waist-worn IMU, 50 Hz", "accelerometer + gyroscope"])
    box(ax, LX, lys[1], LW_, lh, "Windowing", ["2.56 s windows", "50% overlap", "inside one labelled bout"])
    box(ax, LX, lys[2], LW_, lh, "Ten-class task", ["4 basic activities", "6 directional transitions",
                                                   "lie-to-upright = bed exit"])
    cx = LX + LW_ / 2
    arrow(ax, (cx, lys[0]), (cx, lys[1] + lh))
    arrow(ax, (cx, lys[1]), (cx, lys[2] + lh))

    # ---- middle column: four boxes, equal height and equal gap
    mh, mgap = 0.80, 0.15
    mys = [3.20 - i * (mh + mgap) for i in range(4)]
    box(ax, MX, mys[0], MW, mh, "Six-axis hand-crafted", ["accelerometer + gyroscope", "150 features, XGBoost"])
    box(ax, MX, mys[1], MW, mh, "Accelerometer-only", ["hand-crafted, 75 features", "XGBoost"])
    box(ax, MX, mys[2], MW, mh, "Gyroscope-only", ["hand-crafted, 75 features", "XGBoost"])
    box(ax, MX, mys[3], MW, mh, "Foundation model", ["UniMTS, accelerometer only",
                                                           "readouts: linear probe,", "embedding + XGBoost,", "full fine-tuning"])
    mcs = [y + mh / 2 for y in mys]

    # connector 1: task box -> every representation (identical windows and folds)
    bx1, task_c = (LX + LW_ + MX) / 2, lys[2] + lh / 2
    line(ax, (LX + LW_, task_c), (bx1, task_c))
    line(ax, (bx1, mcs[-1]), (bx1, mcs[0]))
    for yc in mcs:
        arrow(ax, (bx1, yc), (MX, yc))

    # ---- right column: evaluation, deployment scoring (equal height)
    rh, rgap = 1.25, 0.95
    ey = 2.75
    dy = ey - rh - rgap
    box(ax, RX, ey, RW, rh, "Evaluation", ["subject-grouped 5-fold CV", "per-subject macro-F1,",
                                          "bed-exit F1,", "label efficiency"])
    box(ax, RX, dy, RW, rh, "Deployment scoring", ["six-axis model only", "calibration, thresholds,",
                                                  "event-level detection,", "boundary windows"])
    # connector 2: every representation -> evaluation
    bx2, ev_c = (MX + MW + RX) / 2, ey + rh / 2
    for yc in mcs:
        line(ax, (MX + MW, yc), (bx2, yc))
    line(ax, (bx2, mcs[-1]), (bx2, max(mcs[0], ev_c)))
    arrow(ax, (bx2, ev_c), (RX, ev_c))
    # evaluation -> deployment scoring
    rcx = RX + RW / 2
    arrow(ax, (rcx, ey), (rcx, dy + rh))

    ax.text(0.05, 0.08, "All representations are scored on the same windows and the same subject-grouped folds.",
            ha="left", va="bottom", fontsize=BODY, color=TEXT)

    FIG.mkdir(exist_ok=True)
    for ext in ("svg", "pdf", "eps", "png", "tif"):
        fig.savefig(FIG / f"fig_pipeline.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved fig_pipeline.{svg,pdf,eps,png,tif} in", FIG)


if __name__ == "__main__":
    main()
