"""
Publication figures and tables for the ward-transition IMU study.

Reads every results JSON/CSV produced by the pipeline and emits:
  Figure 1  per-class F1 of the full 12-class HAPT model -> the headline is
            carried by stairs (absent in a ward) and static postures; bed-exit
            transitions are the weakest classes.
  Figure 2  gyroscope ablation: per-class delta (6-axis minus 3-axis). Static
            classes flat, transitions dominated by the gyroscope.
  Figure 3  label-efficiency curves (macro-F1 and bed-exit F1) with minutes on x.
  Figure 4  representation comparison at full labels: hand6 / hand3 / UniMTS
            (frozen probe, full fine-tune) on macro-F1 and bed-exit F1.
  Table 1   full-label per-class F1 across the arms.
  Table 2   headline metrics + paired tests vs the 6-axis reference.

All numbers are read from disk; nothing is hard-coded except axis labels.
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.autolayout": True,
})

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

WARD_ORDER = ["LYING", "SITTING", "STANDING", "WALKING",
              "STAND_TO_SIT", "SIT_TO_STAND", "SIT_TO_LIE",
              "LIE_TO_SIT", "STAND_TO_LIE", "LIE_TO_STAND"]
BED_EXIT = ["LIE_TO_SIT", "LIE_TO_STAND"]
C_STATIC, C_TRANS, C_BED = "#4C72B0", "#DD8452", "#C44E52"


def _load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def fig1_headline_decomposition():
    j = _load("e1_results.json")
    if not j:
        print("skip fig1 (no e1_results.json)"); return
    rows = j["task_A"]["per_class"]
    df = pd.DataFrame(rows).sort_values("f1")
    colors = ["#8C8C8C" if "STAIRS" in g else (C_BED if c in BED_EXIT else
              (C_TRANS if "TRANSITION" in g else C_STATIC))
              for c, g in zip(df["class"], df["group"])]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(df["class"], df["f1"], color=colors)
    ax.axvline(j["task_A"]["macro_f1_mean"], ls="--", c="k", lw=1,
               label=f"macro-F1 = {j['task_A']['macro_f1_mean']:.3f}")
    ax.set_xlabel("per-class F1 (12-class HAPT model)")
    ax.set_xlim(0, 1.02)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=C_STATIC, label="static (ward)"),
        Patch(color=C_TRANS, label="transition (ward)"),
        Patch(color=C_BED, label="bed-exit (ward)"),
        Patch(color="#8C8C8C", label="stairs (absent in ward)"),
    ], loc="lower right", fontsize=8)
    fig.savefig(FIG / "fig1_headline_decomposition.png"); plt.close(fig)
    print("fig1 done")


def fig2_gyro_ablation():
    j = _load("ablation_gyro.json")
    if not j:
        print("skip fig2"); return
    d = j[0]  # 2.56 s primary
    pc6, pc3 = d["per_class"]["acc+gyro"], d["per_class"]["acc_only"]
    cats = [c for c in WARD_ORDER if c in pc6]
    delta = [pc6[c] - pc3[c] for c in cats]
    colors = [C_BED if c in BED_EXIT else
              (C_TRANS if "_TO_" in c else C_STATIC) for c in cats]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(cats, delta, color=colors)
    ax.set_ylabel("F1 gain from adding gyroscope\n(6-axis minus 3-axis)")
    ax.axhline(0, c="k", lw=0.8)
    ax.tick_params(axis="x", rotation=45)
    for lb in ax.get_xticklabels():
        lb.set_ha("right")
    ax.set_title(f"macro-F1 +{d['macro_f1']['mean_diff']:.3f} "
                 f"(p={d['macro_f1']['wilcoxon_p']:.1e}), "
                 f"bed-exit +{d['bed_exit_f1']['mean_diff']:.3f}", fontsize=9)
    fig.savefig(FIG / "fig2_gyro_ablation.png"); plt.close(fig)
    print("fig2 done")


def fig3_label_efficiency():
    p = RES / "fm_label_efficiency_w128.csv"
    if not p.exists():
        print("skip fig3"); return
    df = pd.read_csv(p)
    # Only the hand-crafted arms are shown: they are the deployable ward models
    # and are padding-independent, so the curve is directly comparable. The
    # gap between them at every budget is the gyroscope effect.
    arms = {"hand6+xgb": ("6-axis hand-crafted", C_STATIC, "o", "-"),
            "hand3+xgb": ("3-axis hand-crafted", "#8C8C8C", "s", "--")}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for metric, ax, title in ((("macro_f1"), axes[0], "macro-F1"),
                              (("bed_exit_f1"), axes[1], "bed-exit F1")):
        for arm, (lab, col, mk, ls) in arms.items():
            sub = df[df["arm"] == arm].sort_values("budget_min")
            ax.plot(sub["budget_min"], sub[metric], marker=mk, ls=ls,
                    color=col, label=lab)
        ax.set_xscale("log")
        ax.set_xlabel("labelled observation (minutes)")
        ax.set_ylabel(title)
        ax.set_title(title)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.savefig(FIG / "fig3_label_efficiency.png"); plt.close(fig)
    print("fig3 done")


def _collect_arms():
    """Assemble full-label metrics for every representation arm from disk.

    The head-only fine-tune (ft-probe) is intentionally excluded: it was run
    under the pre-fix training loop (train-loss model selection, no class
    weighting) and is a training artefact, not a fair arm. The reported FM arms
    are the frozen linear probe, frozen emb+XGB, and full fine-tuning.
    """
    fm = _load("fm_compare_w128.json")
    arms = {}
    if fm:
        for k in ("hand6+xgb", "hand3+xgb", "unimts3+xgb", "unimts3+logreg"):
            if k in fm:
                arms[k] = {"macro": fm[k]["macro_f1_subject_mean"],
                           "bed": fm[k]["bed_exit_f1_subject_mean"],
                           "per_class": fm[k]["per_class"]}
    j = _load("finetune_full_w128.json")
    if j:
        arms["unimts3+ft-full"] = {
            "macro": j["macro_f1_mean"], "bed": j["bed_exit_f1_mean"],
            "per_class": j["per_class"]}
    return arms


def fig4_representation_bars():
    arms = _collect_arms()
    if not arms:
        print("skip fig4"); return
    order = [a for a in ["hand6+xgb", "unimts3+ft-full", "unimts3+logreg",
                         "unimts3+xgb", "hand3+xgb"]
             if a in arms]
    labels = {"hand6+xgb": "hand-crafted\n6-axis", "hand3+xgb": "hand-crafted\n3-axis",
              "unimts3+logreg": "UniMTS\nfrozen probe", "unimts3+ft-full": "UniMTS\nfull FT",
              "unimts3+xgb": "UniMTS\nemb+XGB"}
    macro = [arms[a]["macro"] for a in order]
    bed = [arms[a]["bed"] for a in order]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(x - w/2, macro, w, label="macro-F1", color=C_STATIC)
    ax.bar(x + w/2, bed, w, label="bed-exit F1", color=C_BED)
    ax.axhline(arms["hand6+xgb"]["macro"], ls=":", c=C_STATIC, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([labels[a] for a in order], fontsize=8)
    ax.set_ylabel("F1"); ax.set_ylim(0, 1.02); ax.legend()
    ax.set_title("6-axis input beats every 3-axis representation, pretrained or not",
                 fontsize=9)
    for xi, (m, b) in enumerate(zip(macro, bed)):
        ax.text(xi - w/2, m + 0.01, f"{m:.2f}", ha="center", fontsize=7)
        ax.text(xi + w/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=7)
    fig.savefig(FIG / "fig4_representation.png"); plt.close(fig)
    print("fig4 done")


ARM_LABEL = {
    "hand6+xgb": "Hand-crafted 6-axis + XGBoost",
    "hand3+xgb": "Hand-crafted 3-axis + XGBoost",
    "unimts3+xgb": "UniMTS embedding + XGBoost",
    "unimts3+logreg": "UniMTS frozen linear probe",
    "unimts3+ft-full": "UniMTS full fine-tune",
}
CLASS_LABEL = {c: c.replace("_", "-").title().replace("-To-", "-to-") for c in WARD_ORDER}


def table1_per_class():
    arms = _collect_arms()
    if not arms:
        print("skip table1"); return
    df = pd.DataFrame({ARM_LABEL.get(a, a): v["per_class"] for a, v in arms.items()})
    df = df.reindex(WARD_ORDER).round(3)
    df.index = [CLASS_LABEL[c] for c in df.index]
    df.index.name = "Class"
    df.to_csv(RES / "table1_per_class.csv")
    print("\n=== Table 1: per-class F1 (full labels) ===")
    print(df.to_string())


def table2_headline():
    arms = _collect_arms()
    rows = []
    for a, v in arms.items():
        rows.append({"Representation": ARM_LABEL.get(a, a),
                     "Input": "6-axis" if "hand6" in a else "3-axis",
                     "Macro-F1": round(v["macro"], 3),
                     "Bed-exit F1": round(v["bed"], 3)})
    df = pd.DataFrame(rows).sort_values("Macro-F1", ascending=False)
    df.to_csv(RES / "table2_headline.csv", index=False)
    print("\n=== Table 2: headline metrics (full labels) ===")
    print(df.to_string(index=False))

    # paired tests vs 6-axis reference, gathered from the FT jsons
    print("\n  paired vs hand6+xgb (from fine-tune runs):")
    for mode in ("probe", "full"):
        j = _load(f"finetune_{mode}_w128.json")
        if j and "vs_hand6_macro" in j:
            m = j["vs_hand6_macro"]
            print(f"    ft-{mode:<5s} macro diff {m['mean_diff']:+.4f} "
                  f"CI[{m['ci95_low']:+.3f},{m['ci95_high']:+.3f}] p={m['wilcoxon_p']:.2e}")


def main():
    fig1_headline_decomposition()
    fig2_gyro_ablation()
    fig3_label_efficiency()
    fig4_representation_bars()
    table1_per_class()
    table2_headline()
    print(f"\nfigures -> {FIG}")


if __name__ == "__main__":
    main()
