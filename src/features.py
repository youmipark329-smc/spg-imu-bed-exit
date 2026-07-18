"""Hand-crafted time/frequency features for 6-axis IMU windows.

All features are computed strictly inside a single window, so no statistic
ever crosses a train/test boundary. This makes the feature stage leakage-free
by construction rather than by convention.
"""

import numpy as np
from scipy import stats

FS = 50


def _chan_feats(x: np.ndarray) -> list[np.ndarray]:
    """x: (n, win). Per-channel time + frequency features."""
    jerk = np.diff(x, axis=1)
    fft = np.abs(np.fft.rfft(x, axis=1))[:, 1:]          # drop DC
    psd = fft**2
    psd_n = psd / (psd.sum(axis=1, keepdims=True) + 1e-12)
    freqs = np.fft.rfftfreq(x.shape[1], 1 / FS)[1:]

    return [
        x.mean(1), x.std(1), x.min(1), x.max(1),
        np.median(x, axis=1),
        np.percentile(x, 75, axis=1) - np.percentile(x, 25, axis=1),
        np.abs(x - x.mean(1, keepdims=True)).mean(1),      # MAD
        (x**2).mean(1),                                     # energy
        np.sqrt((x**2).mean(1)),                            # RMS
        stats.skew(x, axis=1),
        stats.kurtosis(x, axis=1),
        (np.diff(np.sign(x - x.mean(1, keepdims=True)), axis=1) != 0).mean(1),  # ZCR
        np.abs(jerk).mean(1), jerk.std(1),                  # jerk
        freqs[np.argmax(psd, axis=1)],                      # dominant frequency
        (psd_n * freqs).sum(1),                             # spectral centroid
        -(psd_n * np.log(psd_n + 1e-12)).sum(1),            # spectral entropy
        psd.sum(1),                                         # spectral energy
    ]


def extract(X: np.ndarray, use_gyro: bool = True) -> np.ndarray:
    """X: (n, win, 6) -> (n, n_features). Channel order: acc xyz, gyro xyz.

    use_gyro=False drops the gyroscope entirely, leaving a 3-axis
    accelerometer-only feature set. The largest wearable motion foundation
    models (RelCon, Inertia-1, LSM) are accelerometer-only, so whether the
    gyroscope carries the sitting/standing signal decides which models are
    usable downstream.
    """
    acc, gyr = X[..., :3], X[..., 3:]
    acc_mag = np.linalg.norm(acc, axis=2)

    chans = [acc[..., i] for i in range(3)] + [acc_mag]
    sigs = [acc]
    if use_gyro:
        chans += [gyr[..., i] for i in range(3)] + [np.linalg.norm(gyr, axis=2)]
        sigs.append(gyr)

    feats: list[np.ndarray] = []
    for c in chans:
        feats.extend(_chan_feats(c))

    # inter-axis correlations: posture is encoded in how gravity splits across
    # the accelerometer axes, so these matter for the low-motion ward classes.
    for sig in sigs:
        for i, j in ((0, 1), (0, 2), (1, 2)):
            a, b = sig[..., i], sig[..., j]
            az = a - a.mean(1, keepdims=True)
            bz = b - b.mean(1, keepdims=True)
            denom = np.sqrt((az**2).sum(1) * (bz**2).sum(1)) + 1e-12
            feats.append((az * bz).sum(1) / denom)

    return np.nan_to_num(np.stack(feats, axis=1).astype(np.float32))
