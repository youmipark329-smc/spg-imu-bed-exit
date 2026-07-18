"""
Ward-10 task: 4 static postures + 6 DIRECTIONAL postural transitions.

The clinical target is bed-exit direction. A monitor must tell LIE_TO_SIT
(patient rising - fall risk) from SIT_TO_LIE (patient settling down). Both are
rare and, in the standard 12-class HAPT setup, they are the two weakest classes.

Reports, per window length:
  - per-class F1
  - full confusion matrix
  - direction-pair confusion: how often a transition is read backwards
  - bed-exit recall / precision

Validation: StratifiedGroupKFold by subject. A subject never spans train/test.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from features import extract
from prepare_hapt import BED_EXIT, DIRECTION_PAIRS, WARD10_NAMES

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SEED = 0
N_FOLDS = 5


def cv_predict(F, y, groups):
    classes = np.unique(y)
    remap = {c: i for i, c in enumerate(classes)}
    yi = np.array([remap[v] for v in y])

    oof = np.full(len(y), -1)
    fold_f1 = []
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(skf.split(F, yi, groups)):
        assert not set(groups[tr]) & set(groups[te]), "subject leaked across folds"
        m = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", random_state=SEED, n_jobs=-1,
        ).fit(F[tr], yi[tr])
        oof[te] = m.predict(F[te])
        f1 = f1_score(yi[te], oof[te], average="macro")
        fold_f1.append(f1)
        print(f"    fold {k + 1}/{N_FOLDS}  macro-F1 {f1:.3f}")
    return yi, oof, classes, np.array(fold_f1)


def analyse(win: int) -> dict:
    d = np.load(RES / f"hapt_windows_w{win}.npz")
    X, yw, subj = d["X"], d["yward10"], d["subj"]
    keep = yw >= 0
    X, yw, subj = X[keep], yw[keep], subj[keep]

    print(f"\n=== window {win} samples ({win / 50:.2f}s) | {X.shape[0]} windows ===")
    F = extract(X)
    yi, oof, classes, folds = cv_predict(F, yw, subj)
    names = [WARD10_NAMES[c] for c in classes]

    per_class = f1_score(yi, oof, average=None)
    tab = pd.DataFrame({
        "class": names,
        "n": [int((yw == c).sum()) for c in classes],
        "f1": [round(float(v), 4) for v in per_class],
    })
    print(f"\n  accuracy {(yi == oof).mean():.4f} | "
          f"macro-F1 {folds.mean():.4f} +/- {folds.std():.4f}")
    print("\n  per-class F1:")
    print(tab.to_string(index=False))

    cm = confusion_matrix(yi, oof)
    cm_df = pd.DataFrame(cm, index=names, columns=names)
    print("\n  confusion matrix (rows = true, cols = predicted):")
    print(cm_df.to_string())

    # direction-pair confusion: the clinically costly error
    print("\n  direction-pair confusion (transition read backwards):")
    pair_rows = []
    for a, b in DIRECTION_PAIRS:
        if a not in names or b not in names:
            continue
        ia, ib = names.index(a), names.index(b)
        n_a, n_b = cm[ia].sum(), cm[ib].sum()
        a_as_b = cm[ia, ib] / n_a if n_a else np.nan
        b_as_a = cm[ib, ia] / n_b if n_b else np.nan
        pair_rows.append({
            "pair": f"{a} <-> {b}",
            f"{a}_read_as_{b}": round(float(a_as_b), 4),
            f"{b}_read_as_{a}": round(float(b_as_a), 4),
        })
        print(f"    {a:>13s} read as {b:<13s} : {a_as_b:.1%}")
        print(f"    {b:>13s} read as {a:<13s} : {b_as_a:.1%}")

    # bed-exit detection: did we catch the patient getting up at all?
    print("\n  bed-exit events (LIE_TO_SIT, LIE_TO_STAND):")
    be_rows = []
    for cls in BED_EXIT:
        if cls not in names:
            continue
        i = names.index(cls)
        rec = cm[i, i] / cm[i].sum() if cm[i].sum() else np.nan
        prec = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() else np.nan
        be_rows.append({"class": cls, "recall": round(float(rec), 4),
                        "precision": round(float(prec), 4)})
        print(f"    {cls:<13s} recall {rec:.3f}  precision {prec:.3f}")

    cm_df.to_csv(RES / f"trans_confusion_w{win}.csv")
    tab.to_csv(RES / f"trans_per_class_w{win}.csv", index=False)

    return {
        "win": win, "win_sec": win / 50, "n_windows": int(X.shape[0]),
        "accuracy": float((yi == oof).mean()),
        "macro_f1_mean": float(folds.mean()), "macro_f1_std": float(folds.std()),
        "fold_macro_f1": folds.tolist(),
        "per_class": tab.to_dict("records"),
        "direction_pairs": pair_rows,
        "bed_exit": be_rows,
        "confusion_matrix": cm.tolist(), "confusion_labels": names,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wins", type=int, nargs="+", default=[64])
    a = p.parse_args()

    out = [analyse(w) for w in a.wins]
    (RES / "transitions_results.json").write_text(json.dumps(out, indent=2))

    if len(out) > 1:
        print("\n\n=== window length sweep ===")
        rows = []
        for r in out:
            trans = [c for c in r["per_class"]
                     if c["class"] not in ("LYING", "SITTING", "STANDING", "WALKING")]
            be = [c for c in r["per_class"] if c["class"] in BED_EXIT]
            rows.append({
                "win_sec": r["win_sec"], "n_windows": r["n_windows"],
                "macro_f1": round(r["macro_f1_mean"], 4),
                "mean_f1_transitions": round(float(np.mean([c["f1"] for c in trans])), 4),
                "mean_f1_bed_exit": round(float(np.mean([c["f1"] for c in be])), 4),
            })
        print(pd.DataFrame(rows).to_string(index=False))

    print(f"\nsaved -> {RES / 'transitions_results.json'}")


if __name__ == "__main__":
    main()
