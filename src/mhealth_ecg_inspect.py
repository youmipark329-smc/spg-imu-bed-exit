"""Sanity-check the MHEALTH ECG: is there a real cardiac signal, and how does it
look during rest vs motion?"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data" / "mhealth_raw" / "MHEALTHDATASET"
FIG = Path(__file__).resolve().parents[1] / "figures"
FS = 50
ACT = {1: "standing", 2: "sitting", 3: "lying", 4: "walking",
       10: "jogging", 11: "running"}


def load(subj):
    return np.loadtxt(DATA / f"mHealth_subject{subj}.log")


def main():
    d = load(1)
    ecg1, ecg2, lab = d[:, 3], d[:, 4], d[:, 23].astype(int)
    print(f"rows {len(d)} | ECG1 range [{ecg1.min():.2f},{ecg1.max():.2f}] mV | "
          f"ECG2 range [{ecg2.min():.2f},{ecg2.max():.2f}] mV")

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for ax, code in zip(axes, (3, 4, 11)):        # lying, walking, running
        seg = ecg1[lab == code]
        t = np.arange(min(len(seg), FS * 6)) / FS  # 6 s
        ax.plot(t, seg[:len(t)], lw=0.7, color="#C44E52")
        ax.set_title(f"{ACT[code]} (label {code})", fontsize=9, loc="left")
        ax.set_ylabel("ECG lead 1 (mV)")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("MHEALTH chest ECG, subject 1 — 6 s per activity", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "mhealth_ecg_inspect.png", dpi=150)
    print(f"saved -> {FIG / 'mhealth_ecg_inspect.png'}")


if __name__ == "__main__":
    main()
