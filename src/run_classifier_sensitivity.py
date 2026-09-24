"""
Is the representation ranking an artefact of choosing XGBoost?

The study fixes the downstream classifier so that only the representation varies.
That design is only meaningful if the ranking it produces does not flip under a
different classifier, so this re-runs the same subject-grouped folds with a
random forest and a scaled multinomial logistic regression as well.

The comparison of interest is within a classifier (six-axis vs three-axis under
the same learner), not across classifiers.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import (cv_oof, make_model, paired_report, per_subject_class_f1,
                      per_subject_macro_f1)
from features import extract
from prepare_hapt import BED_EXIT, WARD10_NAMES

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

REPS = {  # tag -> (use_acc, use_gyro)
    "6ax": (True, True),
    "gyro_only": (False, True),
    "acc_only": (True, False),
}


def rf(seed: int):
    return RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)


def logreg(seed: int):
    # scaling is fitted inside the fold by the pipeline, so no leakage
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed, n_jobs=-1),
    )


CLASSIFIERS = {"xgboost": make_model, "random_forest": rf, "logistic": logreg}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--win", type=int, default=128)
    a = p.parse_args()

    d = np.load(RES / f"hapt_windows_w{a.win}.npz")
    yw, subj = d["yward10"], d["subj"]
    keep = yw >= 0
    X, yw, subj = d["X"][keep], yw[keep], subj[keep]
    print(f"window {a.win} ({a.win / 50:.2f}s) | {X.shape[0]} windows | "
          f"{len(np.unique(subj))} subjects\n")

    feats = {tag: extract(X, use_acc=ua, use_gyro=ug) for tag, (ua, ug) in REPS.items()}

    macro: dict[tuple[str, str], np.ndarray] = {}
    bedexit: dict[tuple[str, str], np.ndarray] = {}
    rows = []
    for cname, mf in CLASSIFIERS.items():
        for rtag, F in feats.items():
            yi, oof, classes = cv_oof(F, yw, subj, model_fn=mf)
            names = [WARD10_NAMES[c] for c in classes]
            _, m = per_subject_macro_f1(yi, oof, subj)
            be_idx = [names.index(c) for c in BED_EXIT if c in names]
            be = np.nanmean(
                np.vstack([per_subject_class_f1(yi, oof, subj, i) for i in be_idx]),
                axis=0)
            macro[(cname, rtag)] = m
            bedexit[(cname, rtag)] = be
            rows.append({"classifier": cname, "representation": rtag,
                         "n_features": F.shape[1],
                         "macro_f1": round(float(m.mean()), 4),
                         "bed_exit_f1": round(float(be.mean()), 4)})
            print(f"  {cname:<14s} {rtag:<10s} macro-F1 {m.mean():.4f}  "
                  f"bed-exit {be.mean():.4f}")

    tab = pd.DataFrame(rows)
    print("\n" + "=" * 66)
    print("macro-F1 by classifier x representation")
    print(tab.pivot(index="classifier", columns="representation",
                    values="macro_f1").to_string())
    print("\nbed-exit F1 by classifier x representation")
    print(tab.pivot(index="classifier", columns="representation",
                    values="bed_exit_f1").to_string())

    out = {"win": a.win, "grid": rows, "paired_6ax_vs_acc_only": {}}
    print("\n" + "=" * 66)
    print("within each classifier: six-axis vs accelerometer-only (paired, n=30)")
    for cname in CLASSIFIERS:
        print(f"\n  [{cname}] macro-F1:")
        out["paired_6ax_vs_acc_only"][cname] = paired_report(
            macro[(cname, "6ax")], macro[(cname, "acc_only")], "6ax", "acc_only")

    (RES / f"classifier_sensitivity_w{a.win}.json").write_text(json.dumps(out, indent=2))
    tab.to_csv(RES / f"classifier_sensitivity_w{a.win}.csv", index=False)
    print(f"\nsaved -> {RES / f'classifier_sensitivity_w{a.win}.json'}")


if __name__ == "__main__":
    main()
