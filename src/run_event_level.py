"""
A1 - event (bout) level bed-exit detection, and debounced false alarms.

Window-level F1 answers "was this 2.56 s slice labelled correctly", which is not
the question a ward alarm asks. A ward asks "was this bed exit flagged at all,
and how often did the device cry wolf". Because a bed-exit bout yields only ~1.5
windows, the two views can differ, so both are reported side by side.

False alarms are merged: consecutive flagged windows outside a bed-exit bout are
one alarm a nurse would answer, not several. This corrects the undebounced
upper bound reported by run_alarm_threshold.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import cv_oof_proba
from features import extract
from prepare_hapt import BED_EXIT, HAPT_NAMES, WARD10_NAMES, load_segments

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FS = 50


def merge_runs(flag: np.ndarray, expid: np.ndarray) -> int:
    """Count maximal runs of consecutive flagged windows within one recording.

    Windows are stored in acquisition order, so a run of consecutive indices in
    the same experiment is one continuous alarm.
    """
    idx = np.flatnonzero(flag)
    if idx.size == 0:
        return 0
    brk = (np.diff(idx) != 1) | (expid[idx[1:]] != expid[idx[:-1]])
    return int(1 + brk.sum())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--win", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)
    a = p.parse_args()

    d = np.load(RES / f"hapt_windows_w{a.win}.npz")
    yw, subj, segid, expid = d["yward10"], d["subj"], d["segid"], d["expid"]
    keep = yw >= 0
    X, yw, subj, segid, expid = (d["X"][keep], yw[keep], subj[keep],
                                 segid[keep], expid[keep])

    F = extract(X)                      # six-axis, deployable configuration
    yi, proba, classes = cv_oof_proba(F, yw, subj)
    names = [WARD10_NAMES[c] for c in classes]
    be_idx = [names.index(c) for c in BED_EXIT if c in names]

    score = proba[:, be_idx].sum(axis=1)
    y_bin = np.isin(yi, be_idx)
    argmax_flag = np.isin(proba.argmax(axis=1), be_idx)

    # bed-exit bouts: those with windows, and the ground-truth total (some bouts
    # were shorter than one window and produced none - counted as missed)
    segs = load_segments()
    be_acts = [k for k, v in HAPT_NAMES.items() if v in BED_EXIT]
    n_bouts_total = int(segs["act"].isin(be_acts).sum())
    be_bouts = np.unique(segid[y_bin])
    n_bouts_windowed = len(be_bouts)
    win_per_hour = 3600 * FS / a.stride
    n_neg = int((~y_bin).sum())

    print(f"window {a.win} ({a.win / FS:.2f}s) | {len(y_bin)} windows | "
          f"bed-exit windows {int(y_bin.sum())}")
    print(f"bed-exit bouts: {n_bouts_windowed} with windows / {n_bouts_total} total "
          f"({n_bouts_total - n_bouts_windowed} shorter than one window)")
    print(f"windows per bed-exit bout: {y_bin.sum() / n_bouts_windowed:.2f}\n")

    rows = []
    for tag, flag in [("argmax", argmax_flag)] + [
            (f"p>={t:.2f}", score >= t) for t in (0.20, 0.35, 0.50, 0.70)]:
        # window level
        tp_w = int((flag & y_bin).sum())
        fp_w = int((flag & ~y_bin).sum())
        sens_w = tp_w / int(y_bin.sum())

        # event level: a bout counts as detected if >=1 (or a majority) of its
        # windows is flagged
        det_any = det_maj = 0
        for b in be_bouts:
            m = segid == b
            if flag[m].any():
                det_any += 1
            if flag[m].mean() > 0.5:
                det_maj += 1

        # false alarms: merge consecutive flagged windows outside bed-exit bouts
        fa_events = merge_runs(flag & ~y_bin, expid)

        rows.append({
            "rule": tag,
            "win_sens": round(sens_w, 4),
            "win_fa_per_h": round(fp_w / n_neg * win_per_hour, 3),
            "evt_sens_any_windowed": round(det_any / n_bouts_windowed, 4),
            "evt_sens_any_all_bouts": round(det_any / n_bouts_total, 4),
            "evt_sens_majority": round(det_maj / n_bouts_windowed, 4),
            "fa_events_per_h": round(fa_events / n_neg * win_per_hour, 3),
            "n_fa_events": fa_events, "n_fp_windows": fp_w,
        })

    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    print("\nevt_sens_any_windowed : >=1 flagged window, over the 114 bouts that produced windows")
    print("evt_sens_any_all_bouts: same, but the 4 too-short bouts counted as missed")
    print("fa_events_per_h       : consecutive flagged windows merged into one alarm")

    out = {"win": a.win, "stride": a.stride,
           "n_bouts_total": n_bouts_total, "n_bouts_windowed": n_bouts_windowed,
           "windows_per_bout": float(y_bin.sum() / n_bouts_windowed),
           "rows": rows}
    (RES / f"event_level_w{a.win}.json").write_text(json.dumps(out, indent=2))
    tab.to_csv(RES / f"event_level_w{a.win}.csv", index=False)
    print(f"\nsaved -> {RES / f'event_level_w{a.win}.json'}")


if __name__ == "__main__":
    main()
