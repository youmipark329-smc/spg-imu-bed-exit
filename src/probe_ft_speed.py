"""Measure CPU forward+backward time to decide if fine-tuning is feasible."""
import sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
UNIMTS = ROOT / "external_UniMTS"
sys.path.insert(0, str(UNIMTS))
from contrastive import ContrastiveModule

torch.set_num_threads(torch.get_num_threads())
print(f"torch threads: {torch.get_num_threads()}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")
args = SimpleNamespace(gyro=0, stft=0, stage="finetune", num_class=10)
model = ContrastiveModule(args)
model.model.load_state_dict(torch.load(UNIMTS / "checkpoint" / "UniMTS.pth",
                                       map_location="cpu"), strict=True)
model = model.to(device)

n_enc = sum(p.numel() for p in model.model.acc.parameters())
n_head = sum(p.numel() for p in model.fc.parameters())
print(f"ST-GCN acc encoder params: {n_enc:,}")
print(f"linear head params:        {n_head:,}")

B = 64
x = torch.randn(B, 3, 200, 22, 1, device=device)
y = torch.randint(0, 10, (B,), device=device)

for mode, train_enc in (("probe (head only)", False), ("full (encoder+head)", True)):
    for p in model.model.acc.parameters():
        p.requires_grad = train_enc
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-4)

    model.train()
    # warmup
    out = model.classifier(x); loss = F.cross_entropy(out, y)
    opt.zero_grad(); loss.backward(); opt.step()
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    N = 5
    for _ in range(N):
        out = model.classifier(x)
        loss = F.cross_entropy(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / N
    print(f"\n{mode}: {dt*1000:.0f} ms / batch(64)  ({B/dt:.0f} windows/s)")

    # project cost: 5 folds, ~6400 train windows/fold, budgets
    per_epoch_batches = 6400 / B
    sec_per_epoch = per_epoch_batches * dt
    print(f"  ~{sec_per_epoch:.1f} s/epoch/fold  |  "
          f"30 epochs x 5 folds = {sec_per_epoch*30*5/60:.1f} min (full-label)")
