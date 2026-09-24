"""
Supplementary figure: raw six-axis signals, unfiltered, as supplied by HAPT.

Shows why no preprocessing is applied. The accelerometer panels sit at clearly
different levels for lying, sitting and standing because the gravity vector
splits differently across the axes with posture - that offset IS the posture
signal, so filtering or per-recording normalisation would remove exactly what
the ward task depends on. The gyroscope panels are near zero while a posture is
held and swing only during the transition, which is the complementary half of
the argument.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from prepare_hapt import HAPT_NAMES, _raw_cache, load_segments

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 600, "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
FS = 50

PANELS = ["LAYING", "SITTING", "STANDING", "LIE_TO_STAND"]
TITLES = {"LAYING": "Lying", "SITTING": "Sitting", "STANDING": "Standing",
          "LIE_TO_STAND": "Lie-to-stand (bed exit)"}
AXCOL = ["#4C72B0", "#DD8452", "#55A868"]
SECONDS = 3.0


def pick(segs, cache, act_name: str, seconds: float):
    """A representative window: the median-length bout of that activity."""
    act = next(k for k, v in HAPT_NAMES.items() if v == act_name)
    g = segs[segs["act"] == act].copy()
    g["dur"] = g["end"] - g["start"]
    s = g.sort_values("dur").iloc[len(g) // 2]
    n = int(seconds * FS)
    mid = (s.start + s.end) // 2
    st = max(s.start, mid - n // 2)
    return cache[s.exp][st : st + n], int(s.user)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="figS_raw_signals.png")
    a = p.parse_args()

    segs = load_segments()
    cache = _raw_cache(segs)

    fig, axes = plt.subplots(2, len(PANELS), figsize=(6.6, 3.7), sharex=True,
                             sharey="row")
    for j, name in enumerate(PANELS):
        w, user = pick(segs, cache, name, SECONDS)
        t = np.arange(len(w)) / FS
        for i, lab in enumerate("xyz"):
            axes[0, j].plot(t, w[:, i], color=AXCOL[i], lw=1.0, label=f"acc {lab}")
            axes[1, j].plot(t, w[:, 3 + i], color=AXCOL[i], lw=1.0, label=f"gyro {lab}")
        axes[0, j].set_title(f"{TITLES[name]}\n(subject {user})", fontsize=8)
        axes[0, j].text(0.03, 0.95, f"({'abcd'[j]})", transform=axes[0, j].transAxes,
                        fontsize=9, fontweight="bold", ha="left", va="top")
        axes[1, j].set_xlabel("time (s)", fontsize=8)

    axes[0, 0].set_ylabel("acceleration (g)\ngravity retained")
    axes[1, 0].set_ylabel("angular velocity\n(rad/s)")
    # one shared legend under the panels, so it never covers a trace or an axis label
    h0, l0 = axes[0, 0].get_legend_handles_labels()
    h1, l1 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="lower center", ncol=6, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = FIG / a.out
    fig.savefig(out)
    plt.close(fig)
    print(f"saved -> {out}")

    # the quantitative claim behind the figure: per-axis accelerometer means
    # separate the three static postures, which is what filtering would destroy
    print("\nmean accelerometer value per axis (g), 3 s excerpt:")
    for name in PANELS:
        w, _ = pick(segs, cache, name, SECONDS)
        print(f"  {TITLES[name]:<26s} "
              f"x {w[:, 0].mean():+.3f}  y {w[:, 1].mean():+.3f}  z {w[:, 2].mean():+.3f}"
              f"   | gyro |.|max {np.abs(w[:, 3:]).max():.3f}")


if __name__ == "__main__":
    main()
