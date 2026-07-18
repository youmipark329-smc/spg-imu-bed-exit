"""Determine how many input channels the released UniMTS checkpoint expects.

contrastive.py builds the IMU encoder as:
    base_channel = 3
    base_channel *= 2 if args.gyro
    base_channel *= 2 if args.stft
    ST_GCN_18(in_channels=base_channel)

so the first ST-GCN conv's in_channels and the data_bn width (22 joints x
channels) together pin down which (gyro, stft) configuration the released
weights were trained under.
"""

from pathlib import Path

import torch

CKPT = Path(__file__).resolve().parents[1] / "external_UniMTS" / "checkpoint" / "UniMTS.pth"

sd = torch.load(CKPT, map_location="cpu")
print(f"checkpoint: {CKPT.name}  ({len(sd)} keys)\n")

conv_w = sd["acc.st_gcn_networks.0.gcn.conv.weight"]
bn_w = sd["acc.data_bn.weight"]
in_ch = conv_w.shape[1]

print(f"acc.st_gcn_networks.0.gcn.conv.weight : {tuple(conv_w.shape)}")
print(f"acc.data_bn.weight                    : {tuple(bn_w.shape)}")
print(f"\ninferred in_channels        = {in_ch}")
print(f"inferred joints x channels  = {bn_w.shape[0]} = 22 x {bn_w.shape[0] // 22}")

print("\nconfiguration table (contrastive.py):")
for gyro in (0, 1):
    for stft in (0, 1):
        b = 3 * (2 if gyro else 1) * (2 if stft else 1)
        hit = "  <== MATCHES RELEASED CHECKPOINT" if b == in_ch else ""
        print(f"  gyro={gyro} stft={stft} -> in_channels={b:2d}, data_bn={22 * b:3d}{hit}")

print("\nacc-encoder keys carrying an input-channel dimension:")
for k, v in sd.items():
    if k.startswith("acc.") and ("data_bn" in k or "st_gcn_networks.0.gcn.conv" in k):
        print(f"  {k:<52s} {tuple(v.shape)}")
