"""M3 fix: paired subject-level test of full fine-tune vs 3-axis hand-crafted.

The representation comparison reported full-FT vs 6-axis with a paired CI/p but
full-FT vs 3-axis only as a point difference. This computes the missing paired
contrast on the same subjects, so both comparisons carry the same statistics.
"""
from pathlib import Path

import numpy as np

from evaluate import cv_oof, paired_report, per_subject_macro_f1
from features import extract
from prepare_hapt import WARD10_NAMES  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

d = np.load(RES / "hapt_windows_w128.npz")
keep = d["yward10"] >= 0
Xraw, yw, subj = d["X"][keep], d["yward10"][keep], d["subj"][keep]

# 3-axis hand-crafted per-subject macro-F1 (same folds as everywhere)
F3 = extract(Xraw, use_gyro=False)
yi3, oof3, _ = cv_oof(F3, yw, subj)
_, macro_hand3 = per_subject_macro_f1(yi3, oof3, subj)

# full fine-tune per-subject macro-F1 from the saved run
import json
ft = json.loads((RES / "finetune_full_w128.json").read_text())
macro_ft = np.array(ft["subject_macro_f1"])

print(f"subjects: hand3={len(macro_hand3)} ft={len(macro_ft)}")
print("\nfull fine-tune vs 3-axis hand-crafted (paired, macro-F1):")
res = paired_report(macro_ft, macro_hand3, "unimts-ft-full", "hand3+xgb")
(RES / "ft_vs_hand3.json").write_text(json.dumps(res, indent=2))
print(f"\nsaved -> {RES / 'ft_vs_hand3.json'}")
