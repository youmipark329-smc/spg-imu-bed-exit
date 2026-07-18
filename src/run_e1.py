"""
E1 - Does the headline HAPT benchmark overstate ward-relevant performance?

Two tasks, one model family (XGBoost on hand-crafted features), subject-grouped CV:

  Task A  standard 12-class HAPT. Report the headline number, then decompose it
          into per-class F1. Same model, no class-count confound: this shows
          *which* classes carry the headline.
  Task B  5-class ward task (LYING/SITTING/STANDING/WALKING/TRANSITION, stairs
          dropped). This is the honest estimate for a ward deployment.

Validation: StratifiedGroupKFold grouped by subject. A subject never appears in
both train and test.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from features import extract
from prepare_hapt import HAPT_NAMES, WARD_MAP, WARD_NAMES

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SEED = 0
N_FOLDS = 5

WARD_RELEVANT_12 = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12]   # everything except stairs
STAIRS_12 = [2, 3]


def model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", device="cuda",
        random_state=SEED, n_jobs=-1,
    )


def cv_predict(F, y, groups, name):
    """Out-of-fold predictions with subject-grouped CV."""
    classes = np.unique(y)
    remap = {c: i for i, c in enumerate(classes)}
    yi = np.array([remap[v] for v in y])

    oof = np.full(len(y), -1)
    fold_f1 = []
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(skf.split(F, yi, groups)):
        assert not set(groups[tr]) & set(groups[te]), "subject leaked across folds"
        m = model().fit(F[tr], yi[tr])
        p = m.predict(F[te])
        oof[te] = p
        f1 = f1_score(yi[te], p, average="macro")
        fold_f1.append(f1)
        print(f"  [{name}] fold {k + 1}/{N_FOLDS}  macro-F1 {f1:.3f}  "
              f"(test subjects: {len(set(groups[te]))})")

    return yi, oof, classes, np.array(fold_f1)


def main() -> None:
    d = np.load(RES / "hapt_windows.npz")
    X, y12, yward, subj = d["X"], d["y12"], d["yward"], d["subj"]

    print("extracting features ...")
    F = extract(X)
    print(f"feature matrix: {F.shape}\n")

    out = {}

    # ---------------- Task A: standard 12-class ----------------
    print("Task A - standard 12-class HAPT")
    yi, oof, classes, folds = cv_predict(F, y12, subj, "12-class")
    acc = (yi == oof).mean()
    per_class = f1_score(yi, oof, average=None)
    print(f"\n  headline accuracy : {acc:.4f}")
    print(f"  macro-F1          : {folds.mean():.4f} +/- {folds.std():.4f}")

    rows = []
    for i, c in enumerate(classes):
        rows.append({
            "class": HAPT_NAMES[c],
            "n": int((y12 == c).sum()),
            "f1": round(float(per_class[i]), 4),
            "group": "STAIRS (absent in ward)" if c in STAIRS_12
                     else WARD_NAMES[WARD_MAP[c]],
        })
    tabA = pd.DataFrame(rows).sort_values("f1", ascending=False)
    print("\n  per-class F1 (same model, no class-count confound):")
    print(tabA.to_string(index=False))

    stairs_f1 = tabA.loc[tabA["class"].isin([HAPT_NAMES[c] for c in STAIRS_12]), "f1"]
    trans_f1 = tabA.loc[tabA["group"] == "TRANSITION", "f1"]
    print(f"\n  mean F1, stairs      : {stairs_f1.mean():.4f}")
    print(f"  mean F1, transitions : {trans_f1.mean():.4f}")

    out["task_A"] = {
        "accuracy": float(acc),
        "macro_f1_mean": float(folds.mean()),
        "macro_f1_std": float(folds.std()),
        "fold_macro_f1": folds.tolist(),
        "per_class": tabA.to_dict("records"),
    }

    # ---------------- Task B: 5-class ward ----------------
    print("\n\nTask B - 5-class ward task (stairs excluded)")
    keep = yward >= 0
    yi_b, oof_b, classes_b, folds_b = cv_predict(
        F[keep], yward[keep], subj[keep], "ward-5"
    )
    accB = (yi_b == oof_b).mean()
    per_class_b = f1_score(yi_b, oof_b, average=None)
    print(f"\n  accuracy : {accB:.4f}")
    print(f"  macro-F1 : {folds_b.mean():.4f} +/- {folds_b.std():.4f}")

    tabB = pd.DataFrame({
        "class": [WARD_NAMES[c] for c in classes_b],
        "n": [int((yward[keep] == c).sum()) for c in classes_b],
        "f1": [round(float(v), 4) for v in per_class_b],
    })
    print("\n  per-class F1:")
    print(tabB.to_string(index=False))

    cm = confusion_matrix(yi_b, oof_b)
    cm_df = pd.DataFrame(cm,
                         index=[WARD_NAMES[c] for c in classes_b],
                         columns=[WARD_NAMES[c] for c in classes_b])
    print("\n  confusion matrix (rows = true, cols = predicted):")
    print(cm_df.to_string())

    out["task_B"] = {
        "accuracy": float(accB),
        "macro_f1_mean": float(folds_b.mean()),
        "macro_f1_std": float(folds_b.std()),
        "fold_macro_f1": folds_b.tolist(),
        "per_class": tabB.to_dict("records"),
        "confusion_matrix": cm.tolist(),
        "confusion_labels": [WARD_NAMES[c] for c in classes_b],
    }

    (RES / "e1_results.json").write_text(json.dumps(out, indent=2))
    tabA.to_csv(RES / "e1_taskA_per_class.csv", index=False)
    tabB.to_csv(RES / "e1_taskB_per_class.csv", index=False)
    cm_df.to_csv(RES / "e1_taskB_confusion.csv")
    print(f"\nsaved -> {RES / 'e1_results.json'}")


if __name__ == "__main__":
    main()
