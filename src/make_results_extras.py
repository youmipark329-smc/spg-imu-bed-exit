"""Additional Results figures and tables.

Two Results subsections carried no visual: the window-length sweep was reported
only as prose, and the gyroscope ablation showed a per-class summary but never
the per-subject paired structure on which every inferential statistic rests.

Emits:
  fig_window_sweep.png   macro-F1, mean transition F1 and bed-exit F1 against
                         window length, on the survivorship-controlled segment
                         set.
  table_window_sweep.csv the same sweep with window counts and the SD across
                         folds.
  fig_gyro_paired.png    per-subject 6-axis vs 3-axis scores, one line per
                         subject, for macro-F1 and bed-exit F1.
  gyro_paired_subjects.csv  the per-subject values behind that figure.

Numbers come from disk or are recomputed with the same functions the reported
analysis used; nothing is hard-coded.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate import cv_oof, per_subject_class_f1, per_subject_macro_f1
from features import extract
from prepare_hapt import BED_EXIT, WARD10_NAMES

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.autolayout": True,
})

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

C6, C3 = "#1f77b4", "#8C8C8C"


def window_sweep():
    p = RES / "window_fair.csv"
    if not p.exists():
        print("skip window sweep (run_window_fair.py first)")
        return
    df = pd.read_csv(p).sort_values("win_sec")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for col, lab, mk in (("macro_f1", "macro-F1 (10 classes)", "o"),
                         ("f1_transitions", "mean F1, six transitions", "s"),
                         ("f1_bed_exit", "bed-exit F1", "^")):
        ax.plot(df["win_sec"], df[col], marker=mk, label=lab)
    ax.set_xticks(df["win_sec"])
    ax.set_xlabel("window length (s)")
    ax.set_ylabel("F1")
    ax.set_ylim(0.5, 0.9)
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(FIG / "fig_window_sweep.png", bbox_inches="tight")
    plt.close(fig)

    out = df.rename(columns={
        "win_sec": "Window length (s)", "n_windows": "Windows",
        "macro_f1": "Macro-F1", "macro_f1_sd": "Macro-F1 SD",
        "f1_transitions": "Mean transition F1", "f1_bed_exit": "Bed-exit F1"})
    out.round(4).to_csv(RES / "table_window_sweep.csv", index=False)
    print("window sweep done")


def gyro_paired(win: int = 128):
    d = np.load(RES / f"hapt_windows_w{win}.npz")
    keep = d["yward10"] >= 0
    X, yw, subj = d["X"][keep], d["yward10"][keep], d["subj"][keep]
    classes = np.unique(yw)
    be_idx = [i for i, c in enumerate(classes) if WARD10_NAMES[c] in BED_EXIT]

    rows = {}
    for tag, use_gyro in (("six", True), ("three", False)):
        F = extract(X, use_gyro=use_gyro)
        yi, oof, _ = cv_oof(F, yw, subj)
        subjects, macro = per_subject_macro_f1(yi, oof, subj)
        be = np.nanmean(np.vstack(
            [per_subject_class_f1(yi, oof, subj, i) for i in be_idx]), axis=0)
        rows[tag] = (subjects, macro, be)
        print(f"  {tag}-axis: macro-F1 {macro.mean():.4f}  bed-exit {np.nanmean(be):.4f}")

    subjects = rows["six"][0]
    df = pd.DataFrame({
        "subject": subjects,
        "macro_f1_six": rows["six"][1], "macro_f1_three": rows["three"][1],
        "bed_exit_f1_six": rows["six"][2], "bed_exit_f1_three": rows["three"][2],
    })
    df.round(4).to_csv(RES / "gyro_paired_subjects.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2))
    panels = (("macro_f1", "macro-F1", axes[0]),
              ("bed_exit_f1", "bed-exit F1", axes[1]))
    for key, title, ax in panels:
        a, b = df[f"{key}_six"].values, df[f"{key}_three"].values
        ok = ~(np.isnan(a) | np.isnan(b))
        for ai, bi in zip(a[ok], b[ok]):
            ax.plot([0, 1], [ai, bi], color="#999", lw=0.8, alpha=0.7,
                    marker="o", ms=3, mfc="white")
        ax.plot([0, 1], [np.nanmean(a), np.nanmean(b)], color=C6, lw=2.4,
                marker="o", ms=6, zorder=5, label="mean")
        worse = int((b[ok] < a[ok]).sum())
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["six-axis", "three-axis"])
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(title)
        ax.set_title(f"{title}\n{worse}/{int(ok.sum())} subjects lower without gyroscope",
                     fontsize=9)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.savefig(FIG / "fig_gyro_paired.png", bbox_inches="tight")
    plt.close(fig)
    print("gyro paired done")


if __name__ == "__main__":
    window_sweep()
    gyro_paired()
