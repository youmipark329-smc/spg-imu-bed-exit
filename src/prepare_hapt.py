"""
HAPT -> windowed 6-axis IMU arrays with ward-relevant class remapping.

Usage:
    python prepare_hapt.py --win 64 --stride 32

Outputs results/hapt_windows_w{WIN}.npz with:
    X       (n, WIN, 6)  acc xyz (g) + gyro xyz (rad/s), 50 Hz
    y12     (n,)         original HAPT activity id 1..12
    yward5  (n,)         coarse ward class, -1 = excluded
    yward10 (n,)         ward class keeping transition DIRECTION, -1 = excluded
    subj    (n,)         user id 1..30
    expid   (n,)         experiment id
    segid   (n,)         index of the source labelled segment

Windows are cut strictly inside one labelled segment, so a window never
straddles two activities and never crosses a subject boundary.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FS = 50

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "hapt_raw" / "RawData"
OUT = ROOT / "results"

HAPT_NAMES = {
    1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS",
    4: "SITTING", 5: "STANDING", 6: "LAYING",
    7: "STAND_TO_SIT", 8: "SIT_TO_STAND", 9: "SIT_TO_LIE",
    10: "LIE_TO_SIT", 11: "STAND_TO_LIE", 12: "LIE_TO_STAND",
}

STAIRS_12 = [2, 3]

# Coarse ward taxonomy: transitions collapsed into one class.
# Kept only as a reference point - it turned out to be an easy task.
WARD5_NAMES = {0: "LYING", 1: "SITTING", 2: "STANDING", 3: "WALKING", 4: "TRANSITION"}
WARD5_MAP = {6: 0, 4: 1, 5: 2, 1: 3,
             7: 4, 8: 4, 9: 4, 10: 4, 11: 4, 12: 4,
             2: -1, 3: -1}

# Primary taxonomy: transition DIRECTION is preserved. Bed-exit monitoring needs
# to tell LIE_TO_SIT (patient rising - fall risk) from SIT_TO_LIE (settling down),
# and these are the two weakest classes in the whole dataset.
WARD10_NAMES = {
    0: "LYING", 1: "SITTING", 2: "STANDING", 3: "WALKING",
    4: "STAND_TO_SIT", 5: "SIT_TO_STAND", 6: "SIT_TO_LIE",
    7: "LIE_TO_SIT", 8: "STAND_TO_LIE", 9: "LIE_TO_STAND",
}
WARD10_MAP = {6: 0, 4: 1, 5: 2, 1: 3,
              7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9,
              2: -1, 3: -1}

# Direction-reversed pairs. Confusion inside a pair is the clinically costly
# error: it means the direction of a bed exit was read backwards.
DIRECTION_PAIRS = [
    ("STAND_TO_SIT", "SIT_TO_STAND"),
    ("SIT_TO_LIE", "LIE_TO_SIT"),
    ("STAND_TO_LIE", "LIE_TO_STAND"),
]

BED_EXIT = ["LIE_TO_SIT", "LIE_TO_STAND"]  # patient getting out of bed


def load_segments() -> pd.DataFrame:
    df = pd.read_csv(
        RAW / "labels.txt", sep=r"\s+", header=None,
        names=["exp", "user", "act", "start", "end"],
    )
    df["n_samples"] = df["end"] - df["start"] + 1
    return df


_CACHE: dict[int, np.ndarray] | None = None


def _raw_cache(segs: pd.DataFrame) -> dict[int, np.ndarray]:
    """Load every experiment's acc+gyro once and reuse across window sizes."""
    global _CACHE
    if _CACHE is None:
        _CACHE = {}
        for exp, user in segs[["exp", "user"]].drop_duplicates().itertuples(index=False):
            acc = np.loadtxt(RAW / f"acc_exp{exp:02d}_user{user:02d}.txt")
            gyr = np.loadtxt(RAW / f"gyro_exp{exp:02d}_user{user:02d}.txt")
            n = min(len(acc), len(gyr))
            _CACHE[exp] = np.hstack([acc[:n], gyr[:n]]).astype(np.float32)
    return _CACHE


def build(win: int, stride: int, verbose: bool = True):
    segs = load_segments()
    cache = _raw_cache(segs)

    X, y12, subj, expid, segid = [], [], [], [], []
    dropped = {a: 0 for a in HAPT_NAMES}
    for si, s in enumerate(segs.itertuples(index=False)):
        seg = cache[s.exp][s.start : s.end + 1]
        if len(seg) < win:
            dropped[s.act] += 1
            continue
        for st in range(0, len(seg) - win + 1, stride):
            X.append(seg[st : st + win])
            y12.append(s.act)
            subj.append(s.user)
            expid.append(s.exp)
            segid.append(si)

    X = np.stack(X)
    y12 = np.asarray(y12, np.int16)
    out = dict(
        X=X, y12=y12,
        yward5=np.asarray([WARD5_MAP[a] for a in y12], np.int8),
        yward10=np.asarray([WARD10_MAP[a] for a in y12], np.int8),
        subj=np.asarray(subj, np.int16),
        expid=np.asarray(expid, np.int16),
        segid=np.asarray(segid, np.int32),
    )

    if verbose:
        print(f"window {win} samples ({win / FS:.2f}s), stride {stride}")
        print(f"  segments dropped (shorter than window): {sum(dropped.values())}")
        for a in sorted(HAPT_NAMES):
            if dropped[a]:
                print(f"    {HAPT_NAMES[a]:<20s} {dropped[a]}")
        print(f"  windows: {X.shape}")
        print("\n  window count by ward-10 class")
        for w in sorted(WARD10_NAMES):
            n = int((out["yward10"] == w).sum())
            mark = "  <- bed exit" if WARD10_NAMES[w] in BED_EXIT else ""
            print(f"    {WARD10_NAMES[w]:<16s} {n:6d}{mark}")
        print(f"    {'(stairs, excl.)':<16s} {int((out['yward10'] == -1).sum()):6d}")

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wins", type=int, nargs="+", default=[64],
                   help="window lengths in samples (50 Hz); stride = 50%% overlap")
    a = p.parse_args()

    OUT.mkdir(exist_ok=True)
    segs = load_segments()
    print(f"labelled segments {len(segs)} | experiments {segs.exp.nunique()} | "
          f"subjects {segs.user.nunique()}\n")

    print("segment duration by activity (seconds)")
    dur = segs.assign(sec=segs.n_samples / FS).groupby("act")["sec"].agg(
        ["count", "min", "median", "max"])
    dur.index = [HAPT_NAMES[i] for i in dur.index]
    print(dur.round(2).to_string())
    print()

    for win in a.wins:
        data = build(win, win // 2)
        path = OUT / f"hapt_windows_w{win}.npz"
        np.savez_compressed(path, **data)
        print(f"\nsaved -> {path}\n")


if __name__ == "__main__":
    main()
