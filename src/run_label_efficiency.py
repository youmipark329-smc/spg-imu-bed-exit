"""
E2 - How many minutes of labelled observation does a ward study need?

This is the feasibility number the planned ward study is built on. A nurse
observes a patient continuously and annotates bouts, so the sampling unit here
is a whole labelled SEGMENT, not a random window: budgets are spent by drawing
observation bouts until the minute budget is exhausted. Random window sampling
would model an observer who annotates disconnected 2.56 s snippets, which nobody
does, and would let 50%-overlapping neighbours pad a small budget.

Sampling is unstratified on purpose. Bed-exit bouts are rare and short, so at a
small budget a realistic observer simply may not witness one - and that, rather
than raw minutes, is what actually binds the ward study.

Test folds are always the full held-out subjects; only the TRAINING budget varies.
Statistics are per-subject (n=30). Primary window = 2.56 s, 6-axis.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from evaluate import make_model, per_subject_class_f1, per_subject_macro_f1
from features import extract
from prepare_hapt import BED_EXIT, WARD10_NAMES, load_segments

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FS = 50
N_FOLDS = 5
BUDGETS_MIN = [2, 5, 10, 20, 40, 80, None]   # None = every labelled minute available
N_SEEDS = 5


def fit_subset(F_tr, y_tr, F_te, seed):
    """Train on whatever classes the budget happened to capture; map back to global ids."""
    present = np.unique(y_tr)
    if len(present) < 2:
        return None, present
    remap = {c: i for i, c in enumerate(present)}
    yi = np.array([remap[v] for v in y_tr])
    m = make_model(seed).fit(F_tr, yi)
    return present[m.predict(F_te)], present


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=int, default=128)
    a = ap.parse_args()

    segs = load_segments()
    seg_sec = (segs.n_samples / FS).values

    d = np.load(RES / f"hapt_windows_w{a.win}.npz")
    yw, subj, segid = d["yward10"], d["subj"], d["segid"]
    keep = yw >= 0
    X, yw, subj, segid = d["X"][keep], yw[keep], subj[keep], segid[keep]

    F = extract(X, use_gyro=True)
    classes_all = np.unique(yw)
    names_all = [WARD10_NAMES[c] for c in classes_all]
    be_names = [c for c in BED_EXIT if c in names_all]

    total_min = seg_sec[np.unique(segid)].sum() / 60
    print(f"window {a.win} ({a.win / FS:.2f}s) | {X.shape[0]} windows | "
          f"{len(np.unique(segid))} segments | {total_min:.1f} labelled minutes total")
    print(f"features {F.shape[1]} | subjects {len(np.unique(subj))}\n")

    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    folds = list(skf.split(F, yw, subj))

    rows = []
    for budget in BUDGETS_MIN:
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(1000 * seed + (budget or 0))
            oof = np.full(len(yw), -1)
            used_min, be_segs = [], []

            for tr, te in folds:
                tr_segs = np.unique(segid[tr])
                if budget is None:
                    chosen = tr_segs
                else:
                    order = rng.permutation(tr_segs)
                    cum = np.cumsum(seg_sec[order]) / 60
                    n_take = int(np.searchsorted(cum, budget) + 1)
                    chosen = order[:n_take]
                used_min.append(seg_sec[chosen].sum() / 60)

                sel = tr[np.isin(segid[tr], chosen)]
                # how many bed-exit bouts did this budget actually witness?
                be_ids = [c for c in classes_all if WARD10_NAMES[c] in be_names]
                be_segs.append(len(np.unique(segid[sel][np.isin(yw[sel], be_ids)])))

                pred, _ = fit_subset(F[sel], yw[sel], F[te], seed)
                oof[te] = pred if pred is not None else yw[sel][0]

            # score per subject, consistent with the rest of the protocol
            _, macro = per_subject_macro_f1(yw, oof, subj)
            idx = {c: i for i, c in enumerate(classes_all)}
            be_idx = [idx[c] for c in classes_all if WARD10_NAMES[c] in be_names]
            yi_g = np.array([idx[v] for v in yw])
            oof_g = np.array([idx.get(v, -1) for v in oof])
            be = np.nanmean(
                np.vstack([per_subject_class_f1(yi_g, oof_g, subj, i) for i in be_idx]),
                axis=0)

            rows.append({
                "budget_min": budget if budget is not None else round(total_min * 0.8, 1),
                "is_full": budget is None,
                "seed": seed,
                "actual_train_min": float(np.mean(used_min)),
                "bed_exit_bouts_seen": float(np.mean(be_segs)),
                "macro_f1": float(macro.mean()),
                "macro_f1_sd": float(macro.std()),
                "bed_exit_f1": float(np.nanmean(be)),
            })
            tag = "ALL" if budget is None else f"{budget:>3d}m"
            print(f"  budget {tag}  seed {seed}  train {np.mean(used_min):6.1f}m  "
                  f"bed-exit bouts {np.mean(be_segs):4.1f}  "
                  f"macro-F1 {macro.mean():.4f}  bed-exit F1 {np.nanmean(be):.4f}")

    df = pd.DataFrame(rows)
    agg = df.groupby("budget_min").agg(
        train_min=("actual_train_min", "mean"),
        bed_exit_bouts=("bed_exit_bouts_seen", "mean"),
        macro_f1=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"),
        bed_exit_f1=("bed_exit_f1", "mean"),
        bed_exit_f1_sd=("bed_exit_f1", "std"),
    ).round(4).reset_index()

    full = agg.iloc[-1]
    agg["pct_of_full_macro"] = (agg.macro_f1 / full.macro_f1 * 100).round(1)
    agg["pct_of_full_bedexit"] = (agg.bed_exit_f1 / full.bed_exit_f1 * 100).round(1)

    print("\n=== label efficiency (mean over 5 seeds, subject-level scoring) ===")
    print(agg.to_string(index=False))

    df.to_csv(RES / "label_efficiency_raw.csv", index=False)
    agg.to_csv(RES / "label_efficiency.csv", index=False)
    (RES / "label_efficiency.json").write_text(
        json.dumps({"total_labelled_min": float(total_min),
                    "window": a.win,
                    "agg": agg.to_dict("records")}, indent=2))
    print(f"\nsaved -> {RES / 'label_efficiency.csv'}")


if __name__ == "__main__":
    main()
