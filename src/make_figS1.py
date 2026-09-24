"""Figure S1: macro-F1 against bed-exit F1, one point per representation arm.

Redrawn at revision 1 so that the gyroscope-only arm added then is included.
Values are read from the reported headline table (analysis/tables/
table2_headline.csv, the same numbers as Table 4 of the main text), so the
figure cannot drift from the manuscript. Output: figures/figS1_representation_scatter.png
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams.update({"figure.dpi": 150, "savefig.dpi": 600, "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
# The headline table is written by make_figures.py into results/; the manuscript
# copy one level up is used only when this repository sits beside the manuscript tree.
_CANDIDATES = [ROOT / "results" / "table2_headline.csv",
               ROOT.parent / "analysis" / "tables" / "table2_headline.csv"]
TABLE = next((c for c in _CANDIDATES if c.exists()), _CANDIDATES[0])

C_SIX, C_ACC, C_GYRO = "#4C72B0", "#C44E52", "#c2571a"
SHORT = {
    "Hand-crafted 6-axis + XGBoost": "hand-crafted 6-axis",
    "UniMTS full fine-tune": "UniMTS full fine-tune",
    "UniMTS frozen linear probe": "UniMTS frozen probe",
    "Hand-crafted 3-axis gyroscope-only + XGBoost": "hand-crafted gyroscope-only",
    "UniMTS embedding + XGBoost": "UniMTS emb + XGBoost",
    "Hand-crafted 3-axis accelerometer-only + XGBoost": "hand-crafted accelerometer-only",
}
# label offsets (dx, dy, ha) chosen so that no label overlaps a point
OFF = {
    "hand-crafted 6-axis": (-0.012, 0.016, "right"),
    "UniMTS full fine-tune": (0.010, 0.012, "left"),
    "UniMTS frozen probe": (0.010, 0.012, "left"),
    "hand-crafted gyroscope-only": (-0.012, -0.030, "right"),
    "UniMTS emb + XGBoost": (0.010, -0.024, "left"),
    "hand-crafted accelerometer-only": (0.010, 0.010, "left"),
}


def main():
    df = pd.read_csv(TABLE)
    fig, ax = plt.subplots(figsize=(5.6, 4.1))   # placed at 5.6 in
    six = df[df["Input"].str.startswith("6-axis")].iloc[0]
    ax.axvspan(six["Macro-F1"], 1, color="#EAF0F7", zorder=0)
    ax.axhspan(six["Bed-exit F1"], 1, color="#EAF0F7", zorder=0)
    for _, r in df.iterrows():
        name = SHORT[r["Representation"]]
        inp = r["Input"]
        is6, isg = inp.startswith("6-axis"), "gyroscope" in inp
        ax.scatter(r["Macro-F1"], r["Bed-exit F1"], s=150 if is6 else 95,
                   marker="D" if is6 else ("s" if isg else "o"),
                   color=C_SIX if is6 else "white",
                   edgecolor=C_SIX if is6 else (C_GYRO if isg else C_ACC), lw=1.8, zorder=3)
        dx, dy, ha = OFF[name]
        ax.annotate(f"{name}\n({inp})", (r["Macro-F1"] + dx, r["Bed-exit F1"] + dy),
                    ha=ha, fontsize=7.5, color="#222", zorder=4)
    ax.set_xlim(0.60, 0.90); ax.set_ylim(0.30, 0.75)
    ax.set_xlabel("macro-F1 (ten-class ward task)"); ax.set_ylabel("bed-exit F1")
    ax.text(0.895, 0.735, "six-axis territory:\nunreached by any\nthree-axis arm",
            ha="right", va="top", fontsize=7.5, color=C_SIX, style="italic")
    ax.grid(alpha=0.25, zorder=0)
    fig.tight_layout()
    fig.savefig(FIG / "figS1_representation_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print("saved", FIG / "figS1_representation_scatter.png")


if __name__ == "__main__":
    main()
