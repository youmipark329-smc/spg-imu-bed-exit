"""
Does fine-tuning UniMTS close the gap that the frozen probe left open?

The frozen linear probe reached macro-F1 0.708 / bed-exit 0.485 - it beat the
3-axis hand-crafted baseline but lost badly to 6-axis hand-crafted (0.823 /
0.658). Fine-tuning is the last thing a reviewer will ask for, so we run it.

Two modes, both from the released 3-channel checkpoint:
  probe  train the linear head only (encoder frozen)
  full   train ST-GCN encoder + head end-to-end

Same subject-grouped folds and per-subject statistics as every other arm. This
does NOT change the fundamental constraint - the released checkpoint ingests 3
channels, so fine-tuning cannot recover the gyroscope information. It only tests
whether adapting the pretrained weights beats reading them frozen.

GPU required (CPU-measured at 8 windows/s for full mode -> ~32 h).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
UNIMTS = ROOT / "external_UniMTS"
sys.path.insert(0, str(UNIMTS))
sys.path.insert(0, str(ROOT / "src"))

from contrastive import ContrastiveModule          # noqa: E402
from evaluate import (paired_report, per_subject_class_f1,      # noqa: E402
                      per_subject_macro_f1)
from prepare_hapt import BED_EXIT, WARD10_NAMES     # noqa: E402
from unimts_embed import preprocess                 # noqa: E402

N_FOLDS = 5
SEED = 0


def build_model(num_class, device):
    args = SimpleNamespace(gyro=0, stft=0, stage="finetune", num_class=num_class)
    m = ContrastiveModule(args)
    m.model.load_state_dict(torch.load(UNIMTS / "checkpoint" / "UniMTS.pth",
                                       map_location="cpu"), strict=True)
    return m.to(device)


def _predict(model, X, device, batch):
    model.eval()
    preds = []
    with torch.no_grad():
        Xt = torch.from_numpy(X).float()
        for i in range(0, len(Xt), batch):
            preds.append(model.classifier(Xt[i:i + batch].to(device)).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def train_fold(Xtr, ytr, gtr, Xte, mode, num_class, device, epochs, batch, lr,
               seed=SEED):
    """Fair fine-tuning: inner-validation model selection + class weighting.

    - An inner validation split by SUBJECT (never the test subjects) selects the
      epoch on validation macro-F1, not training loss. Selecting on train loss
      picks the most over-fit epoch and collapses the rare transition classes
      (that is why the first attempt scored below the frozen probe).
    - Class-weighted cross-entropy so the rare bed-exit transitions are not
      drowned by the abundant static classes.
    - ``seed`` controls weight-init/batch order and the inner-val subject draw,
      so repeated runs with different seeds quantify fine-tuning variance.
    """
    from sklearn.metrics import f1_score
    torch.manual_seed(seed)

    # inner-val: hold out ~20% of the training SUBJECTS
    rng = np.random.default_rng(seed)
    subs = np.unique(gtr)
    n_val = max(1, int(round(0.2 * len(subs))))
    val_subs = set(rng.choice(subs, size=n_val, replace=False).tolist())
    va = np.array([g in val_subs for g in gtr])
    tr = ~va
    Xin, yin = Xtr[tr], ytr[tr]
    Xva, yva = Xtr[va], ytr[va]

    model = build_model(num_class, device)
    if mode == "probe":
        for p in model.model.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    cnt = np.bincount(yin, minlength=num_class).astype(float)
    w = (cnt.sum() / (num_class * np.clip(cnt, 1, None)))
    cw = torch.tensor(w, dtype=torch.float32, device=device)

    Xin_t = torch.from_numpy(Xin).float()
    yin_t = torch.from_numpy(yin).long()
    n = len(Xin_t)

    best_state, best_val = None, -1.0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            out = model.classifier(Xin_t[idx].to(device))
            loss = F.cross_entropy(out.float(), yin_t[idx].to(device), weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()
        val_f1 = f1_score(yva, _predict(model, Xva, device, batch), average="macro")
        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return _predict(model, Xte, device, batch)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=int, default=128)
    ap.add_argument("--mode", choices=["probe", "full"], default="full")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--smoke", action="store_true",
                    help="run only the first fold to sanity-check the training loop")
    ap.add_argument("--padding", type=int, default=64,
                    help="resampled sequence length; 64 covers a 2.56s window (avoids 4x wrap-tiling)")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="seed for the CV split, inner-val draw, and training; "
                         "vary it across runs to quantify fine-tuning variance")
    ap.add_argument("--tag", type=str, default="",
                    help="output filename suffix, e.g. _seed1, so multi-seed runs "
                         "do not overwrite each other or the canonical file")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | mode: {a.mode} | epochs {a.epochs} | seed {a.seed}")
    if device == "cpu":
        print("WARNING: CPU fine-tuning is ~10-32 h; aborting. Install CUDA torch.")
        return

    d = np.load(RES / f"hapt_windows_w{a.win}.npz")
    keep = d["yward10"] >= 0
    Xraw, yw, subj = d["X"][keep], d["yward10"][keep], d["subj"][keep]

    classes = np.unique(yw)
    remap = {c: i for i, c in enumerate(classes)}
    yi = np.array([remap[v] for v in yw])
    names = [WARD10_NAMES[c] for c in classes]

    print(f"preprocessing to UniMTS input (3-channel, padding={a.padding}) ...")
    X = preprocess(Xraw, 50, gyro=0, padding=a.padding).numpy()   # (N, 3, padding, 22, 1)

    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=a.seed)
    oof = np.full(len(yi), -1)
    t0 = time.time()
    from sklearn.metrics import f1_score
    for k, (tr, te) in enumerate(skf.split(X, yi, subj)):
        assert not set(subj[tr]) & set(subj[te])
        p = train_fold(X[tr], yi[tr], subj[tr], X[te], a.mode, len(classes), device,
                       a.epochs, a.batch, a.lr, seed=a.seed)
        oof[te] = p
        print(f"  fold {k+1}/{N_FOLDS}  macro-F1 {f1_score(yi[te], p, average='macro'):.4f}"
              f"  ({time.time()-t0:.0f}s elapsed)")
        if a.smoke:
            print("  [smoke] stopping after fold 1")
            return

    _, macro = per_subject_macro_f1(yi, oof, subj)
    be_idx = [names.index(c) for c in BED_EXIT if c in names]
    be = np.nanmean(np.vstack([per_subject_class_f1(yi, oof, subj, i) for i in be_idx]),
                    axis=0)
    from sklearn.metrics import f1_score
    per = f1_score(yi, oof, average=None)

    print(f"\n=== UniMTS fine-tune ({a.mode}) ===")
    print(f"subject-mean macro-F1 : {macro.mean():.4f} +/- {macro.std():.4f}")
    print(f"subject-mean bed-exit : {np.nanmean(be):.4f}")
    print("\nper-class F1:")
    for nm, v in zip(names, per):
        print(f"  {nm:<14s} {v:.4f}")

    out = {
        "mode": a.mode, "epochs": a.epochs, "seed": a.seed,
        "macro_f1_mean": float(macro.mean()), "macro_f1_std": float(macro.std()),
        "bed_exit_f1_mean": float(np.nanmean(be)),
        "per_class": {nm: float(v) for nm, v in zip(names, per)},
        "subject_macro_f1": macro.tolist(),
        "subject_bed_exit_f1": [None if np.isnan(x) else float(x) for x in be],
    }

    # paired subject-level test vs the 6-axis hand-crafted arm (the one to beat)
    from features import extract
    from evaluate import cv_oof
    F6 = extract(Xraw, use_gyro=True)
    yi6, oof6, cls6 = cv_oof(F6, yw, subj)
    _, macro6 = per_subject_macro_f1(yi6, oof6, subj)
    be6 = np.nanmean(np.vstack(
        [per_subject_class_f1(yi6, oof6, subj, list(cls6).index(c))
         for c in cls6 if WARD10_NAMES[c] in BED_EXIT]), axis=0)
    print("\n  fine-tuned UniMTS vs 6-axis hand-crafted (paired, macro-F1):")
    out["vs_hand6_macro"] = paired_report(macro, macro6, f"unimts-ft-{a.mode}", "hand6+xgb")
    print("\n  fine-tuned UniMTS vs 6-axis hand-crafted (paired, bed-exit F1):")
    out["vs_hand6_bedexit"] = paired_report(be, be6, f"unimts-ft-{a.mode}", "hand6+xgb")

    out_path = RES / f"finetune_{a.mode}_w{a.win}{a.tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {out_path}")
    print("\nreference (from fm_compare):")
    print("  frozen probe        macro 0.7082  bed-exit 0.4852")
    print("  3-axis hand-crafted macro 0.6467  bed-exit 0.3512")
    print("  6-axis hand-crafted macro 0.8225  bed-exit 0.6575")


if __name__ == "__main__":
    main()
