"""
A2 - what boundary-crossing windows cost.

The main analysis places every window strictly inside one annotated bout, which a
sliding window in a ward cannot do: it will straddle the moment a transition
starts. This slides windows continuously across the contiguous stretches of HAPT
(642 adjacent segment pairs, 176 of them touching a bed-exit class) and scores
the held-out subjects on that stream instead.

HAPT is not fully annotated - most segment pairs are separated by unlabelled gaps
of 0.1-54 s - so a genuinely continuous 24 h stream cannot be simulated without
inventing labels for those gaps. The scope here is therefore the boundary effect
only, which is what the reviewer asked about; full streaming is left to the
prospective ward study.

Training uses the same within-bout windows as the main analysis; only the test
stream changes, so the drop isolates the deployment shift.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from evaluate import N_FOLDS, SEED, make_model, per_subject_class_f1, per_subject_macro_f1
from features import extract
from prepare_hapt import (BED_EXIT, WARD10_MAP, WARD10_NAMES, _raw_cache,
                          load_segments)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FS = 50


def build_continuous(win: int, stride: int):
    """Slide windows over every contiguous labelled stretch, boundaries included."""
    segs = load_segments()
    cache = _raw_cache(segs)

    Xs, y_centre, y_major, is_bnd, subj, expid = [], [], [], [], [], []
    for exp, g in segs.groupby("exp"):
        raw = cache[exp]
        lab = np.full(len(raw), -1, np.int16)
        user = int(g["user"].iloc[0])
        for s in g.itertuples(index=False):
            lab[s.start : s.end + 1] = s.act

        # a maximal run of labelled samples is exactly a contiguous stretch
        labelled = lab >= 0
        edges = np.flatnonzero(np.diff(labelled.astype(np.int8)))
        starts = ([0] if labelled[0] else []) + list(edges[labelled[edges + 1]] + 1)
        ends = list(edges[labelled[edges]] + 1) + ([len(lab)] if labelled[-1] else [])

        for a, b in zip(starts, ends):
            for st in range(a, b - win + 1, stride):
                seg_lab = lab[st : st + win]
                uniq, cnt = np.unique(seg_lab, return_counts=True)
                Xs.append(raw[st : st + win])
                y_centre.append(int(seg_lab[win // 2]))
                y_major.append(int(uniq[cnt.argmax()]))
                is_bnd.append(len(uniq) > 1)
                subj.append(user)
                expid.append(int(exp))

    return (np.stack(Xs), np.asarray(y_centre), np.asarray(y_major),
            np.asarray(is_bnd), np.asarray(subj), np.asarray(expid))


def score_block(yi, pred, subj, be_idx, mask) -> dict:
    """Per-subject macro-F1 and bed-exit F1 restricted to a subset of windows."""
    if mask.sum() == 0:
        return {}
    y, p, s = yi[mask], pred[mask], subj[mask]
    _, macro = per_subject_macro_f1(y, p, s)
    be = np.nanmean(np.vstack([per_subject_class_f1(y, p, s, i) for i in be_idx]),
                    axis=0)
    y_bin, p_bin = np.isin(y, be_idx), np.isin(p, be_idx)
    return {
        "n_windows": int(mask.sum()),
        "macro_f1": float(np.nanmean(macro)),
        "bed_exit_f1": float(np.nanmean(be)),
        "bed_exit_sens": float((p_bin & y_bin).sum() / max(1, y_bin.sum())),
        "bed_exit_windows": int(y_bin.sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--win", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--label", choices=["centre", "majority"], default="centre")
    a = p.parse_args()

    # training set: the within-bout windows of the main analysis
    d = np.load(RES / f"hapt_windows_w{a.win}.npz")
    keep_tr = d["yward10"] >= 0
    Xtr, ytr, str_ = d["X"][keep_tr], d["yward10"][keep_tr], d["subj"][keep_tr]

    Xc, yc_centre, yc_major, is_bnd, subj_c, _ = build_continuous(a.win, a.stride)
    y_raw = yc_centre if a.label == "centre" else yc_major
    yw_c = np.array([WARD10_MAP[v] for v in y_raw])
    keep_te = yw_c >= 0
    Xc, yw_c, is_bnd, subj_c = Xc[keep_te], yw_c[keep_te], is_bnd[keep_te], subj_c[keep_te]

    print(f"window {a.win} ({a.win / FS:.2f}s) stride {a.stride} | label rule: {a.label}")
    print(f"training windows (within-bout): {len(ytr)}")
    print(f"continuous test windows: {len(yw_c)}  "
          f"({int(is_bnd.sum())} boundary-crossing, {int((~is_bnd).sum())} pure)\n")

    classes = np.unique(ytr)
    remap = {c: i for i, c in enumerate(classes)}
    yi_tr = np.array([remap[v] for v in ytr])
    yi_te = np.array([remap[v] for v in yw_c])
    names = [WARD10_NAMES[c] for c in classes]
    be_idx = [names.index(c) for c in BED_EXIT if c in names]

    Ftr, Fte = extract(Xtr), extract(Xc)

    pred = np.full(len(yi_te), -1)
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, _ in skf.split(Ftr, yi_tr, str_):
        test_subjects = np.setdiff1d(np.unique(str_), np.unique(str_[tr]))
        m = np.isin(subj_c, test_subjects)
        assert not set(str_[tr]) & set(subj_c[m]), "subject leaked across folds"
        pred[m] = make_model(SEED).fit(Ftr[tr], yi_tr[tr]).predict(Fte[m])
    assert (pred >= 0).all(), "some continuous windows were never scored"

    blocks = {
        "pure_only": ~is_bnd,
        "boundary_only": is_bnd,
        "all_continuous": np.ones(len(is_bnd), bool),
    }
    res = {k: score_block(yi_te, pred, subj_c, be_idx, m) for k, m in blocks.items()}
    tab = pd.DataFrame(res).T
    print(tab.to_string())

    pure, allc = res["pure_only"], res["all_continuous"]
    print(f"\ndrop from pure to full continuous stream: "
          f"macro-F1 {pure['macro_f1']:.4f} -> {allc['macro_f1']:.4f} "
          f"({allc['macro_f1'] - pure['macro_f1']:+.4f}) | "
          f"bed-exit F1 {pure['bed_exit_f1']:.4f} -> {allc['bed_exit_f1']:.4f} "
          f"({allc['bed_exit_f1'] - pure['bed_exit_f1']:+.4f})")

    out = {"win": a.win, "stride": a.stride, "label_rule": a.label,
           "n_boundary": int(is_bnd.sum()), "n_pure": int((~is_bnd).sum()),
           "blocks": res}
    (RES / f"boundary_windows_w{a.win}_{a.label}.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {RES / f'boundary_windows_w{a.win}_{a.label}.json'}")


if __name__ == "__main__":
    main()
