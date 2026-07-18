"""
Extract UniMTS embeddings for our HAPT ward windows.

UniMTS (NeurIPS 2024) is a CLIP-style contrastive model: a ST-GCN-18 over a
22-joint SMPL skeleton graph is aligned with CLIP text embeddings. We use its
IMU encoder as a frozen feature extractor and then push the 512-d embeddings
through the SAME protocol as the hand-crafted features (subject-grouped CV,
per-subject statistics, minute-budgeted label efficiency). That keeps the
comparison honest: only the representation changes.

Preprocessing replicates data.load_custom_data exactly:
  1. acc must be m/s^2  -- HAPT ships g, so we convert (x 9.80665)
  2. scatter our single sensor into its skeleton joint; other joints stay zero
     (this is UniMTS's intended use - their own TNDA-HAR run fills 5 of 22)
  3. resample original_sampling_rate -> 20 Hz
  4. 'wrap' pad to padding_size=200 (=10 s)
  5. reshape to (N, C, T, V, M=1) for ST-GCN

NOTE on wrap padding: upstream builds a mask but never passes it to the model
(evaluate_custom.py moves it to cuda and drops it), so a 2.56 s window is tiled
~4x and the model sees all of it as real signal. We keep upstream behaviour
rather than silently diverging from it, and record it as a limitation.

HAPT is a single waist-mounted phone -> SMPL joint 0 (pelvis).
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.signal import resample

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
UNIMTS = ROOT / "external_UniMTS"

G_TO_MS2 = 9.80665
N_JOINTS = 22
WAIST_JOINT = 0          # SMPL pelvis, per joint_assignment.png
TARGET_HZ = 20
PADDING = 200            # 10 s at 20 Hz, UniMTS default


def preprocess(X: np.ndarray, src_hz: int, gyro: int, padding: int = PADDING) -> torch.Tensor:
    """(N, T, 6) g/rad-per-s -> (N, C, T, V, M) ready for ST-GCN.

    gyro=0 reproduces upstream's channel drop (evaluate_custom.py): the loader
    builds 6 channels per joint and the model then keeps only the accelerometer
    triplet of each joint. The released checkpoint is 3-channel, so gyro=0 is
    the only setting its weights actually fit.

    padding: sequence length after resampling. Upstream defaults to 200 (=10 s
    at 20 Hz), but a 2.56 s window is only ~51 samples, so 200 wrap-tiles the
    real signal ~4x. That 4x is wasted compute and, under back-prop, 4x the
    activation memory (it drove the full fine-tune near-OOM). ST-GCN pools
    globally over time, so a shorter padding that just covers the real window is
    both cheaper and free of the tiling artefact. Frozen-probe extraction and
    fine-tuning must use the SAME padding to stay comparable.
    """
    X = X.astype(np.float64).copy()
    X[..., :3] *= G_TO_MS2                       # HAPT accelerometer is in g

    n, t, _ = X.shape
    allX = np.zeros((n, t, N_JOINTS, 6))
    allX[:, :, WAIST_JOINT] = X
    allX = allX.reshape(n, t, N_JOINTS * 6)

    new_len = int(t / src_hz * TARGET_HZ)
    allX = np.array([resample(s, new_len) for s in allX])

    if allX.shape[1] < padding:
        allX = np.pad(allX, ((0, 0), (0, padding - allX.shape[1]), (0, 0)), "wrap")
    allX = allX[:, :padding, :]

    if not gyro:
        c = allX.shape[-1]
        idx = np.array([range(i, i + 3) for i in range(0, c, 6)]).flatten()
        allX = allX[:, :, idx]

    ch = 6 if gyro else 3
    x = torch.from_numpy(allX).float()
    return x.reshape(len(x), padding, N_JOINTS, ch).permute(0, 3, 1, 2).unsqueeze(-1)


def load_model(ckpt: Path, device: str, gyro: int):
    import sys
    sys.path.insert(0, str(UNIMTS))
    from contrastive import ContrastiveModule

    # Only the IMU encoder is used, so the CLIP text tower is never called and a
    # CPU run is fine for frozen extraction.
    args = SimpleNamespace(gyro=gyro, stft=0, stage="evaluate", num_class=10)
    model = ContrastiveModule(args)
    state = torch.load(ckpt, map_location="cpu")
    missing, unexpected = model.model.load_state_dict(state, strict=False)
    vis = [k for k in missing if not k.startswith(("transformer.", "token_embedding",
                                                  "positional_embedding", "ln_final",
                                                  "text_projection", "logit_scale"))]
    print(f"checkpoint loaded | missing={len(missing)} unexpected={len(unexpected)}")
    if vis:
        print(f"  WARNING missing non-text keys (first 5): {vis[:5]}")
    return model.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=int, default=128)
    ap.add_argument("--src_hz", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--checkpoint", type=str,
                    default=str(UNIMTS / "checkpoint" / "UniMTS.pth"))
    ap.add_argument("--gyro", type=int, default=0,
                    help="0 matches the released checkpoint (3-channel, acc only)")
    ap.add_argument("--padding", type=int, default=64,
                    help="resampled sequence length; 64 covers a 2.56s window at 20Hz")
    a = ap.parse_args()

    d = np.load(RES / f"hapt_windows_w{a.win}.npz")
    keep = d["yward10"] >= 0
    X = d["X"][keep]
    print(f"windows {X.shape} | acc range (g) [{X[..., :3].min():.2f}, "
          f"{X[..., :3].max():.2f}] | gyro range [{X[..., 3:].min():.2f}, "
          f"{X[..., 3:].max():.2f}]")

    inp = preprocess(X, a.src_hz, a.gyro, a.padding)
    print(f"model input {tuple(inp.shape)}  (N, C, T, V, M) | gyro={a.gyro} | padding={a.padding}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model = load_model(Path(a.checkpoint), device, a.gyro)
    embs = []
    with torch.no_grad():
        for i in range(0, len(inp), a.batch):
            b = inp[i : i + a.batch].to(device)
            embs.append(model.encode_image(b).cpu().numpy())
            if i % (a.batch * 40) == 0:
                print(f"  {i}/{len(inp)}")
    E = np.concatenate(embs).astype(np.float32)
    print(f"embeddings {E.shape}")

    out = RES / f"unimts_emb_w{a.win}.npz"
    np.savez_compressed(out, E=E, yward10=d["yward10"][keep], subj=d["subj"][keep],
                        segid=d["segid"][keep])
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
