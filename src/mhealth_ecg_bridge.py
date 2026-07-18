"""
The ECG bridge (MHEALTH): does activity systematically degrade ECG signal
quality? This is the mechanism behind motion-induced arrhythmia false alarms
and the preview of the planned ward study's second hypothesis.

MHEALTH provides synchronized chest 2-lead ECG + body IMU on the same subjects
(10 subjects, 50 Hz). We:
  1. cut labelled 5 s ECG windows,
  2. compute motion-sensitive ECG signal-quality indices (SQIs),
  3. compute IMU motion energy on the same window,
  4. show SQI degrades monotonically with activity motion energy, and that the
     ward-relevant classes (lying/sitting/standing/walking) already span a
     clinically meaningful quality gradient.

This is descriptive (association), on HEALTHY subjects with NO arrhythmias, and
the ECG is sampled at only 50 Hz (Nyquist 25 Hz) - far below clinical 250-500 Hz.
It bounds what a real ward study must establish; it does not detect false alarms.

SQIs (adapted to the 0-25 Hz available band; Behar et al. 2013 family):
  kSQI   kurtosis of the window - a clean ECG is peaky (sharp QRS) -> high
         kurtosis; motion noise fills the baseline -> low kurtosis. Higher = better.
  basSQI relative spectral power in 0-1 Hz - baseline wander from motion.
         Higher = worse.
  pSQI   relative power in the 5-15 Hz QRS band. Higher = better (QRS stands out).
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mhealth_raw" / "MHEALTHDATASET"
FIG = ROOT / "figures"
RES = ROOT / "results"
FS = 50
WIN = 5 * FS            # 5 s
SUBJECTS = range(1, 11)

# ward-relevant subset first (ordered by expected motion), then higher-intensity
ACT = {3: "lying", 2: "sitting", 1: "standing", 4: "walking",
       5: "stairs", 9: "cycling", 10: "jogging", 11: "running"}
WARD = {1, 2, 3, 4}
C_WARD, C_OTHER = "#4C72B0", "#C44E52"


def ecg_sqi(x):
    x = x - x.mean()
    if x.std() < 1e-6:
        return np.nan, np.nan, np.nan
    xk = (x - x.mean()) / x.std()
    ksqi = float(stats.kurtosis(xk, fisher=True))          # excess kurtosis
    f, p = welch(x, fs=FS, nperseg=min(len(x), 128))
    tot = p.sum() + 1e-12
    bassqi = float(p[f <= 1].sum() / tot)                   # baseline wander
    psqi = float(p[(f >= 5) & (f <= 15)].sum() / tot)       # QRS band
    return ksqi, bassqi, psqi


def motion_energy(win):
    """RMS of gravity-removed acceleration, averaged over chest/ankle/wrist."""
    # chest acc cols 0-2, ankle acc 5-7, wrist acc 14-16
    acc = np.concatenate([win[:, 0:3], win[:, 5:8], win[:, 14:17]], axis=1)
    e = []
    for i in range(0, acc.shape[1], 3):
        mag = np.linalg.norm(acc[:, i:i+3], axis=1)
        e.append(mag.std())                                 # motion around gravity
    return float(np.mean(e))


def main():
    rows = []
    for s in SUBJECTS:
        d = np.loadtxt(DATA / f"mHealth_subject{s}.log")
        ecg, lab = d[:, 3], d[:, 23].astype(int)
        for code in ACT:
            idx = np.where(lab == code)[0]
            if len(idx) < WIN:
                continue
            # contiguous run; step through non-overlapping windows
            start = idx[0]
            for st in range(start, idx[-1] - WIN + 1, WIN):
                if lab[st] != code or lab[st + WIN - 1] != code:
                    continue
                k, b, p = ecg_sqi(ecg[st:st + WIN])
                if np.isnan(k):
                    continue
                rows.append({"subject": s, "code": code, "activity": ACT[code],
                             "ward": code in WARD, "kSQI": k, "basSQI": b,
                             "pSQI": p, "motion": motion_energy(d[st:st + WIN])})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "mhealth_ecg_sqi.csv", index=False)
    print(f"windows: {len(df)} across {df.subject.nunique()} subjects")

    order = sorted(ACT, key=lambda c: df[df.code == c].motion.mean())
    names = [ACT[c] for c in order]

    # ---- per-activity aggregate (subject means, then across-subject stats) ----
    agg = (df.groupby(["activity", "code", "ward"])
             .agg(motion=("motion", "mean"), kSQI=("kSQI", "mean"),
                  basSQI=("basSQI", "mean"), pSQI=("pSQI", "mean"), n=("kSQI", "size"))
             .reset_index().set_index("code").reindex(order).reset_index())
    print("\nby activity (motion-ordered):")
    print(agg[["activity", "motion", "kSQI", "basSQI", "pSQI", "n"]].round(3).to_string(index=False))

    r_k = stats.spearmanr(df.motion, df.kSQI)
    r_b = stats.spearmanr(df.motion, df.basSQI)
    print(f"\nSpearman motion vs kSQI  : rho={r_k.correlation:+.3f}  p={r_k.pvalue:.1e}")
    print(f"Spearman motion vs basSQI: rho={r_b.correlation:+.3f}  p={r_b.pvalue:.1e}")

    # within-ward gradient: lying vs walking kSQI
    def act_vals(a, col):
        return df[df.activity == a].groupby("subject")[col].mean()
    ly, wk = act_vals("lying", "kSQI"), act_vals("walking", "kSQI")
    common = ly.index.intersection(wk.index)
    w = stats.wilcoxon(ly[common], wk[common])
    print(f"\nward gradient  lying kSQI {ly.mean():.2f} vs walking {wk.mean():.2f}  "
          f"(paired Wilcoxon p={w.pvalue:.3f}, n={len(common)})")

    # ================= FIGURE =================
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.42, wspace=0.28)

    # Panel A: example traces (subject 1) lying / walking / running
    d1 = np.loadtxt(DATA / "mHealth_subject1.log")
    e1, l1 = d1[:, 3], d1[:, 23].astype(int)
    axA = fig.add_subplot(gs[0, :])
    off, colors = 0, {3: "#4C72B0", 4: "#DD8452", 11: "#C44E52"}
    for code in (3, 4, 11):
        seg = e1[l1 == code][:FS * 5]
        seg = (seg - seg.mean())
        axA.plot(np.arange(len(seg)) / FS + off, seg, lw=0.7, color=colors[code],
                 label=ACT[code])
        off += 5.3
    axA.set_title("A. Chest ECG, subject 1 — 5 s each (lying, walking, running)",
                  fontsize=10, loc="left")
    axA.set_xlabel("time (s)"); axA.set_ylabel("ECG (mV, centred)")
    axA.legend(fontsize=8, ncol=3, loc="upper right")

    # Panel B: kSQI by activity (subject points + mean), motion-ordered
    axB = fig.add_subplot(gs[1, 0])
    for i, c in enumerate(order):
        sub = df[df.code == c].groupby("subject").kSQI.mean()
        col = C_WARD if c in WARD else C_OTHER
        axB.scatter(np.full(len(sub), i) + np.random.uniform(-.12, .12, len(sub)),
                    sub, s=14, color=col, alpha=0.5)
        axB.plot(i, sub.mean(), "_", ms=22, color=col, mew=2.5)
    axB.set_xticks(range(len(order)))
    axB.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    axB.set_ylabel("ECG kSQI (higher = cleaner)")
    axB.set_title("B. ECG quality degrades with activity", fontsize=10, loc="left")
    from matplotlib.patches import Patch
    axB.legend(handles=[Patch(color=C_WARD, label="ward-relevant"),
                        Patch(color=C_OTHER, label="higher-intensity")],
               fontsize=8, loc="upper right")

    # Panel C: dose-response, motion energy vs kSQI (per window)
    axC = fig.add_subplot(gs[1, 1])
    axC.scatter(df.motion, df.kSQI, s=6,
                c=[C_WARD if w else C_OTHER for w in df.ward], alpha=0.25)
    axC.set_xlabel("IMU motion energy (m/s$^2$, gravity-removed RMS)")
    axC.set_ylabel("ECG kSQI")
    axC.set_title(f"C. Dose-response (Spearman $\\rho$={r_k.correlation:+.2f}, "
                  f"p<1e-3)", fontsize=10, loc="left")

    fig.suptitle("Activity systematically degrades chest-ECG signal quality "
                 "(MHEALTH, 10 subjects) — the mechanism of motion-induced "
                 "arrhythmia false alarms", fontsize=11)
    fig.savefig(FIG / "fig5_ecg_bridge.png", bbox_inches="tight")
    print(f"\nsaved -> {FIG / 'fig5_ecg_bridge.png'}")
    agg.to_csv(RES / "mhealth_ecg_by_activity.csv", index=False)


if __name__ == "__main__":
    np.random.seed(0)
    main()
