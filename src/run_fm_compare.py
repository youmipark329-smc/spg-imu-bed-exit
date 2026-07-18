"""
Does UniMTS pretraining buy anything over hand-crafted features?

Same windows, same subject-grouped folds, same subject-level statistics - only
the representation changes. Three arms:

  hand-crafted + XGBoost   our baseline (B3)
  UniMTS emb   + XGBoost   same classifier, pretrained representation
  UniMTS emb   + logreg    linear probe, the standard way to read a frozen FM

The question that matters for the ward study is not peak accuracy - hand-crafted
features already reach macro-F1 0.82 with every label. It is whether pretraining
reaches that level from FEWER labelled minutes, because minutes of nurse
observation are the real constraint. So the label-efficiency curve is run for
every arm, not just the pooled score.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from evaluate import (make_model, paired_report, per_subject_class_f1,
                      per_subject_macro_f1)
from features import extract
from prepare_hapt import BED_EXIT, WARD10_NAMES, load_segments

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FS = 50
N_FOLDS = 5
BUDGETS_MIN = [5, 10, 20, 40, 80, None]
N_SEEDS = 3


def fit_predict(kind, F_tr, y_tr, F_te, seed):
    present = np.unique(y_tr)
    if len(present) < 2:
        return None
    remap = {c: i for i, c in enumerate(present)}
    yi = np.array([remap[v] for v in y_tr])
    if kind == "logreg":
        sc = StandardScaler().fit(F_tr)
        m = LogisticRegression(max_iter=3000, C=1.0, random_state=seed)
        m.fit(sc.transform(F_tr), yi)
        return present[m.predict(sc.transform(F_te))]
    m = make_model(seed).fit(F_tr, yi)
    return present[m.predict(F_te)]


def score_oof(yw, oof, subj, classes_all):
    _, macro = per_subject_macro_f1(yw, oof, subj)
    idx = {c: i for i, c in enumerate(classes_all)}
    be_idx = [idx[c] for c in classes_all if WARD10_NAMES[c] in BED_EXIT]
    yi_g = np.array([idx[v] for v in yw])
    oof_g = np.array([idx.get(v, -1) for v in oof])
    be = np.nanmean(
        np.vstack([per_subject_class_f1(yi_g, oof_g, subj, i) for i in be_idx]), axis=0)
    return macro, be


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=int, default=128)
    ap.add_argument("--quick", action="store_true",
                    help="full-label comparison only; skip label-efficiency curves")
    a = ap.parse_args()

    d = np.load(RES / f"hapt_windows_w{a.win}.npz")
    keep = d["yward10"] >= 0
    X, yw, subj, segid = d["X"][keep], d["yward10"][keep], d["subj"][keep], d["segid"][keep]

    emb_path = RES / f"unimts_emb_w{a.win}.npz"
    e = np.load(emb_path)
    E = e["E"]
    assert np.array_equal(e["yward10"], yw) and np.array_equal(e["subj"], subj), \
        "embedding rows are not aligned with the window arrays"
    print(f"windows {X.shape} | UniMTS emb {E.shape} | subjects {len(np.unique(subj))}")

    F_hand6 = extract(X, use_gyro=True)
    F_hand3 = extract(X, use_gyro=False)
    classes_all = np.unique(yw)

    # The released UniMTS checkpoint is 3-channel (accelerometer only), verified
    # by check_unimts_ckpt.py: in_channels=3, data_bn=66=22x3, and it loads with
    # zero missing/unexpected keys at gyro=0. So hand3 is the like-for-like
    # comparison that isolates what pretraining buys, while hand6 is what a ward
    # could actually deploy with the gyroscope the FM cannot read.
    arms = {
        "hand6+xgb": ("xgb", F_hand6),
        "hand3+xgb": ("xgb", F_hand3),
        "unimts3+xgb": ("xgb", E),
        "unimts3+logreg": ("logreg", E),
    }
    REF = "hand3+xgb"        # same input as UniMTS -> isolates pretraining
    CHALLENGERS = ["unimts3+xgb", "unimts3+logreg", "hand6+xgb"]

    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    folds = list(skf.split(F_hand6, yw, subj))

    # ---------- full-label comparison ----------
    print("\n=== full labels ===")
    res_full, subj_scores = {}, {}
    for name, (kind, F) in arms.items():
        oof = np.full(len(yw), -1)
        for tr, te in folds:
            oof[te] = fit_predict(kind, F[tr], yw[tr], F[te], 0)
        macro, be = score_oof(yw, oof, subj, classes_all)
        subj_scores[name] = (macro, be)
        pooled = f1_score(yw, oof, average=None, labels=classes_all)
        res_full[name] = {
            "macro_f1_subject_mean": float(macro.mean()),
            "macro_f1_subject_sd": float(macro.std()),
            "bed_exit_f1_subject_mean": float(np.nanmean(be)),
            "per_class": {WARD10_NAMES[c]: float(v)
                          for c, v in zip(classes_all, pooled)},
        }
        print(f"  {name:<16s} macro-F1 {macro.mean():.4f} +/- {macro.std():.4f}  "
              f"bed-exit {np.nanmean(be):.4f}")

    print(f"\n  paired over subjects vs {REF}, macro-F1:")
    for ch in CHALLENGERS:
        print(f"  -- {ch}")
        res_full[f"{ch}_vs_{REF}_macro"] = paired_report(
            subj_scores[ch][0], subj_scores[REF][0], ch, REF)
    print(f"\n  paired over subjects vs {REF}, bed-exit F1:")
    for ch in CHALLENGERS:
        print(f"  -- {ch}")
        res_full[f"{ch}_vs_{REF}_bedexit"] = paired_report(
            subj_scores[ch][1], subj_scores[REF][1], ch, REF)

    print("\n  can the accel-only FM close the gap to 6-axis hand-crafted?")
    for ch in ("unimts3+xgb", "unimts3+logreg"):
        print(f"  -- {ch} vs hand6+xgb (macro-F1)")
        res_full[f"{ch}_vs_hand6_macro"] = paired_report(
            subj_scores[ch][0], subj_scores["hand6+xgb"][0], ch, "hand6+xgb")
        print(f"  -- {ch} vs hand6+xgb (bed-exit)")
        res_full[f"{ch}_vs_hand6_bedexit"] = paired_report(
            subj_scores[ch][1], subj_scores["hand6+xgb"][1], ch, "hand6+xgb")

    tab = pd.DataFrame({k: v["per_class"] for k, v in res_full.items()
                        if "per_class" in v})
    print("\n  pooled per-class F1:")
    print(tab.to_string())
    tab.to_csv(RES / f"fm_per_class_w{a.win}.csv")
    (RES / f"fm_compare_w{a.win}.json").write_text(json.dumps(res_full, indent=2))

    if a.quick:
        print("\n[quick] skipping label-efficiency curves")
        return

    # ---------- label efficiency, every arm ----------
    segs = load_segments()
    seg_sec = (segs.n_samples / FS).values
    print("\n=== label efficiency ===")
    rows = []
    for name, (kind, F) in arms.items():
        for budget in BUDGETS_MIN:
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(1000 * seed + (budget or 0))
                oof = np.full(len(yw), -1)
                for tr, te in folds:
                    tr_segs = np.unique(segid[tr])
                    if budget is None:
                        chosen = tr_segs
                    else:
                        order = rng.permutation(tr_segs)
                        cum = np.cumsum(seg_sec[order]) / 60
                        chosen = order[: int(np.searchsorted(cum, budget) + 1)]
                    sel = tr[np.isin(segid[tr], chosen)]
                    p = fit_predict(kind, F[sel], yw[sel], F[te], seed)
                    oof[te] = p if p is not None else yw[sel][0]
                macro, be = score_oof(yw, oof, subj, classes_all)
                rows.append({
                    "arm": name,
                    "budget_min": budget if budget else 156.4,
                    "seed": seed,
                    "macro_f1": float(macro.mean()),
                    "bed_exit_f1": float(np.nanmean(be)),
                })
            m = np.mean([r["macro_f1"] for r in rows[-N_SEEDS:]])
            b = np.mean([r["bed_exit_f1"] for r in rows[-N_SEEDS:]])
            print(f"  {name:<16s} budget {str(budget or 'ALL'):>4s}m  "
                  f"macro-F1 {m:.4f}  bed-exit {b:.4f}")

    df = pd.DataFrame(rows)
    piv = df.groupby(["arm", "budget_min"])[["macro_f1", "bed_exit_f1"]].mean().round(4)
    print("\n=== label efficiency summary ===")
    print(piv.to_string())
    df.to_csv(RES / f"fm_label_efficiency_raw_w{a.win}.csv", index=False)
    piv.reset_index().to_csv(RES / f"fm_label_efficiency_w{a.win}.csv", index=False)
    (RES / f"fm_compare_w{a.win}.json").write_text(json.dumps(res_full, indent=2))
    print(f"\nsaved -> {RES / f'fm_compare_w{a.win}.json'}")


if __name__ == "__main__":
    main()
