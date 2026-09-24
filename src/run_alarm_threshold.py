"""
Bed-exit alarm: probability calibration and the missed-alarm / false-alarm trade-off.

A ward alarm fires on a predicted probability, not on argmax, so macro-F1 alone
cannot specify a deployable alarm policy. This reports, for the deployable
six-axis hand-crafted model:

  * a threshold sweep (sensitivity, specificity, precision, F1) for the binary
    bed-exit question, scored as P(LIE_TO_SIT) + P(LIE_TO_STAND);
  * window-level false alarms per hour, the quantity a ward actually budgets for;
  * calibration (Brier score + reliability curve), which argmax metrics hide.

All predictions are out-of-fold under the same subject-grouped folds as the main
analysis, so no subject contributes to its own score.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)

from evaluate import cv_oof_proba
from features import extract
from prepare_hapt import BED_EXIT, WARD10_NAMES

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FS = 50


def sweep(y_bin: np.ndarray, score: np.ndarray, win: int, stride: int) -> pd.DataFrame:
    """Threshold sweep with false alarms expressed per hour of monitoring."""
    # one window is emitted every `stride` samples, so this many windows per hour
    win_per_hour = 3600 * FS / stride
    n_neg = int((~y_bin).sum())

    rows = []
    for t in np.round(np.arange(0.05, 1.00, 0.05), 2):
        pred = score >= t
        tp = int((pred & y_bin).sum())
        fp = int((pred & ~y_bin).sum())
        fn = int((~pred & y_bin).sum())
        sens = tp / (tp + fn) if tp + fn else np.nan
        prec = tp / (tp + fp) if tp + fp else np.nan
        spec = 1 - fp / n_neg if n_neg else np.nan
        f1 = 2 * prec * sens / (prec + sens) if prec and sens else 0.0
        rows.append({
            "threshold": t, "sensitivity": sens, "specificity": spec,
            "precision": prec, "f1": f1,
            "false_alarms_per_hour": fp / n_neg * win_per_hour,
            "tp": tp, "fp": fp, "fn": fn,
        })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--win", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)
    a = p.parse_args()

    d = np.load(RES / f"hapt_windows_w{a.win}.npz")
    yw, subj = d["yward10"], d["subj"]
    keep = yw >= 0
    X, yw, subj = d["X"][keep], yw[keep], subj[keep]

    F = extract(X)  # six-axis, the deployable configuration
    yi, proba, classes = cv_oof_proba(F, yw, subj)
    names = [WARD10_NAMES[c] for c in classes]

    be_idx = [names.index(c) for c in BED_EXIT if c in names]
    score = proba[:, be_idx].sum(axis=1)          # P(any bed-exit transition)
    y_bin = np.isin(yi, be_idx)

    print(f"window {a.win} ({a.win / FS:.2f}s) stride {a.stride} | "
          f"{len(y_bin)} windows | bed-exit positives {int(y_bin.sum())}")
    print(f"bed-exit classes: {[names[i] for i in be_idx]}\n")

    auroc = roc_auc_score(y_bin, score)
    auprc = average_precision_score(y_bin, score)
    brier = brier_score_loss(y_bin, score)
    prevalence = float(y_bin.mean())
    print(f"AUROC {auroc:.4f} | AUPRC {auprc:.4f} "
          f"(prevalence {prevalence:.4f}) | Brier {brier:.5f}\n")

    tab = sweep(y_bin, score, a.win, a.stride)
    print(tab.to_string(index=False,
                        float_format=lambda v: f"{v:.4f}"))

    # Why the binary numbers beat the per-class ones: collapsing the two bed-exit
    # classes removes exactly the lie-to-sit <-> lie-to-stand confusion the paper
    # reports. Quantify it rather than assume it.
    pred = proba.argmax(axis=1)
    print("\nper-class (10-class argmax) vs binary bed-exit detection:")
    for i in be_idx:
        m = yi == i
        rec = float((pred[m] == i).mean())
        as_other_be = float(np.isin(pred[m], [j for j in be_idx if j != i]).mean())
        print(f"    {names[i]:<14s} n={int(m.sum()):3d}  "
              f"recall(exact class) {rec:.3f}  "
              f"predicted as the other bed-exit class {as_other_be:.3f}  "
              f"recall(either bed-exit) {rec + as_other_be:.3f}")
    print(f"    binary recall at argmax: "
          f"{float(np.isin(pred[y_bin], be_idx).mean()):.3f}")

    # Reliability: with 2% prevalence, quantile bins collapse to ~0, so use uniform
    # bins over the score range and report bin counts so empty bins are visible.
    frac_pos, mean_pred = calibration_curve(y_bin, score, n_bins=10, strategy="uniform")
    edges = np.linspace(0, 1, 11)
    counts = np.histogram(score, bins=edges)[0]
    print("\nreliability curve (uniform bins):")
    for lo, hi, n in zip(edges[:-1], edges[1:], counts):
        if n == 0:
            continue
        m = (score >= lo) & (score < hi if hi < 1 else score <= 1)
        print(f"    [{lo:.1f},{hi:.1f})  n={int(n):5d}  "
              f"mean predicted {score[m].mean():.4f}   observed {y_bin[m].mean():.4f}")
    print(f"\ncalibration-in-the-large: mean predicted {score.mean():.5f} "
          f"vs observed {y_bin.mean():.5f}")

    out = {
        "win": a.win, "win_sec": a.win / FS, "stride": a.stride,
        "n_windows": int(len(y_bin)), "n_bed_exit_windows": int(y_bin.sum()),
        "prevalence": prevalence,
        "auroc": float(auroc), "auprc": float(auprc), "brier": float(brier),
        "sweep": tab.to_dict(orient="records"),
        "reliability": {"mean_predicted": mean_pred.tolist(),
                        "observed_fraction": frac_pos.tolist()},
    }
    (RES / f"alarm_threshold_w{a.win}.json").write_text(json.dumps(out, indent=2))
    tab.to_csv(RES / f"alarm_threshold_w{a.win}.csv", index=False)
    print(f"\nsaved -> {RES / f'alarm_threshold_w{a.win}.json'}")


if __name__ == "__main__":
    main()
