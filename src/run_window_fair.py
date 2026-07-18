"""
Is the long-window advantage real, or survivorship?

The naive sweep scores longer windows higher. But a 2.56 s window cannot be cut
from a segment shorter than 2.56 s, so that condition silently discards 40
segments - including 29 of 62 SIT_TO_STAND (47%) - and then reports a score on
the survivors. The short, fast transitions that get dropped are exactly the hard
ones.

This script re-runs the sweep restricted to the segments that survive at every
window length tested, so all conditions are scored on the SAME events. Any
remaining gap is a genuine window-length effect.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from features import extract
from prepare_hapt import BED_EXIT, WARD10_NAMES, load_segments

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SEED = 0
N_FOLDS = 5
WINS = [32, 64, 96, 128]
STATIC = {"LYING", "SITTING", "STANDING", "WALKING"}


def score(F, y, groups):
    classes = np.unique(y)
    remap = {c: i for i, c in enumerate(classes)}
    yi = np.array([remap[v] for v in y])
    oof = np.full(len(y), -1)
    folds = []
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in skf.split(F, yi, groups):
        assert not set(groups[tr]) & set(groups[te])
        m = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", random_state=SEED, n_jobs=-1,
        ).fit(F[tr], yi[tr])
        oof[te] = m.predict(F[te])
        folds.append(f1_score(yi[te], oof[te], average="macro"))
    return yi, oof, classes, np.array(folds)


def main() -> None:
    segs = load_segments()
    # segments long enough to yield at least one window at every tested length
    common = set(np.where(segs.n_samples.values >= max(WINS))[0])
    print(f"segments total          : {len(segs)}")
    print(f"segments in common set  : {len(common)}")
    print(f"segments excluded       : {len(segs) - len(common)}")

    rows = []
    for win in WINS:
        d = np.load(RES / f"hapt_windows_w{win}.npz")
        yw, subj, segid = d["yward10"], d["subj"], d["segid"]
        keep = (yw >= 0) & np.isin(segid, list(common))
        X = d["X"][keep]
        yw, subj = yw[keep], subj[keep]

        F = extract(X)
        yi, oof, classes, folds = score(F, yw, subj)
        names = [WARD10_NAMES[c] for c in classes]
        per = f1_score(yi, oof, average=None)
        trans = [v for n, v in zip(names, per) if n not in STATIC]
        be = [v for n, v in zip(names, per) if n in BED_EXIT]

        rows.append({
            "win_sec": win / 50,
            "n_windows": int(len(yw)),
            "macro_f1": round(float(folds.mean()), 4),
            "macro_f1_sd": round(float(folds.std()), 4),
            "f1_transitions": round(float(np.mean(trans)), 4),
            "f1_bed_exit": round(float(np.mean(be)), 4),
        })
        print(f"  win {win / 50:.2f}s  n={len(yw):6d}  "
              f"macro-F1 {folds.mean():.4f}  trans {np.mean(trans):.4f}  "
              f"bed-exit {np.mean(be):.4f}")

    tab = pd.DataFrame(rows)
    print("\n=== fair sweep: identical segments at every window length ===")
    print(tab.to_string(index=False))
    tab.to_csv(RES / "window_fair.csv", index=False)
    (RES / "window_fair.json").write_text(json.dumps(rows, indent=2))
    print(f"\nsaved -> {RES / 'window_fair.csv'}")


if __name__ == "__main__":
    main()
