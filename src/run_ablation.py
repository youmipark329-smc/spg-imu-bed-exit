"""
E3 - Does the gyroscope carry the bed-exit signal?

This decides which foundation models are usable downstream. The largest wearable
motion FMs (RelCon 1B segments, Inertia-1 18.2M hours, Google LSM 40M hours) are
accelerometer-only: they would discard half of a 6-axis recording. Only the much
smaller UniMTS and LIMU-BERT ingest accel+gyro.

  gyro helps          -> stuck with the small 6-axis FMs
  gyro does not help  -> free to use the large accel-only FMs

All statistics are paired over SUBJECTS (n=30), not folds. See evaluate.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import (cv_oof, paired_report, per_subject_class_f1,
                      per_subject_macro_f1)
from features import extract
from prepare_hapt import BED_EXIT, WARD10_NAMES

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
STATIC = {"LYING", "SITTING", "STANDING", "WALKING"}


def run_one(win: int) -> dict:
    d = np.load(RES / f"hapt_windows_w{win}.npz")
    yw, subj = d["yward10"], d["subj"]
    keep = yw >= 0
    X, yw, subj = d["X"][keep], yw[keep], subj[keep]
    print(f"\n{'=' * 66}\nwindow {win} samples ({win / 50:.2f}s) | "
          f"{X.shape[0]} windows | {len(np.unique(subj))} subjects\n{'=' * 66}")

    res: dict = {"win": win, "win_sec": win / 50, "n_windows": int(X.shape[0])}
    subj_macro, subj_bedexit, per_class = {}, {}, {}

    for tag, use_gyro in (("acc+gyro", True), ("acc_only", False)):
        F = extract(X, use_gyro=use_gyro)
        yi, oof, classes = cv_oof(F, yw, subj)
        names = [WARD10_NAMES[c] for c in classes]

        subjects, macro = per_subject_macro_f1(yi, oof, subj)
        subj_macro[tag] = macro

        be_idx = [names.index(c) for c in BED_EXIT if c in names]
        be = np.nanmean(
            np.vstack([per_subject_class_f1(yi, oof, subj, i) for i in be_idx]), axis=0
        )
        subj_bedexit[tag] = be

        # pooled per-class F1, for the descriptive table only
        from sklearn.metrics import f1_score
        per_class[tag] = dict(zip(names, np.round(f1_score(yi, oof, average=None), 4)))

        print(f"  {tag:<10s} n_features={F.shape[1]:3d}  "
              f"subject-mean macro-F1 {macro.mean():.4f} +/- {macro.std():.4f}")

    print("\n  paired over subjects - macro-F1:")
    res["macro_f1"] = paired_report(subj_macro["acc+gyro"], subj_macro["acc_only"],
                                    "acc+gyro", "acc_only")
    print("\n  paired over subjects - bed-exit F1 (LIE_TO_SIT, LIE_TO_STAND):")
    res["bed_exit_f1"] = paired_report(subj_bedexit["acc+gyro"], subj_bedexit["acc_only"],
                                       "acc+gyro", "acc_only")

    tab = pd.DataFrame(per_class)
    tab["delta"] = (tab["acc+gyro"] - tab["acc_only"]).round(4)
    print("\n  pooled per-class F1:")
    print(tab.to_string())
    tab.to_csv(RES / f"ablation_gyro_w{win}.csv")

    res["per_class"] = {k: {kk: float(vv) for kk, vv in v.items()}
                        for k, v in per_class.items()}
    res["subject_macro_f1"] = {k: v.tolist() for k, v in subj_macro.items()}
    res["subject_bed_exit_f1"] = {k: v.tolist() for k, v in subj_bedexit.items()}
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wins", type=int, nargs="+", default=[128, 64],
                   help="first is the primary analysis window")
    a = p.parse_args()

    out = [run_one(w) for w in a.wins]
    (RES / "ablation_gyro.json").write_text(json.dumps(out, indent=2))

    print(f"\n\n{'=' * 66}\nsummary (subject-level, n=30)\n{'=' * 66}")
    rows = [{
        "win_sec": r["win_sec"],
        "macro_f1_6ax": round(r["macro_f1"]["mean_acc+gyro"], 4),
        "macro_f1_3ax": round(r["macro_f1"]["mean_acc_only"], 4),
        "diff": round(r["macro_f1"]["mean_diff"], 4),
        "ci95": f"[{r['macro_f1']['ci95_low']:+.3f}, {r['macro_f1']['ci95_high']:+.3f}]",
        "p": f"{r['macro_f1']['wilcoxon_p']:.2e}",
        "bedexit_6ax": round(r["bed_exit_f1"]["mean_acc+gyro"], 4),
        "bedexit_3ax": round(r["bed_exit_f1"]["mean_acc_only"], 4),
    } for r in out]
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nsaved -> {RES / 'ablation_gyro.json'}")


if __name__ == "__main__":
    main()
