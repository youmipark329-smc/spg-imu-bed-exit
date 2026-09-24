"""
Shared evaluation with the SUBJECT as the unit of analysis.

Fold-level statistics (n=5) are not a valid basis for inference: a two-sided
Wilcoxon on 5 pairs cannot return p < 2/2^5 = 0.0625 no matter how large the
effect. Every comparison here is therefore paired over subjects (n=30), which is
also the unit a ward study would report.

Each subject sits in exactly one test fold under StratifiedGroupKFold, so the
out-of-fold predictions give every subject exactly one held-out score.
"""

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

N_FOLDS = 5
SEED = 0


def make_model(seed: int = SEED) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=seed, n_jobs=-1,
    )


def cv_oof(F, y, groups, seed: int = SEED, n_folds: int = N_FOLDS, model_fn=None):
    """Out-of-fold predictions with subject-grouped CV.

    Returns (yi, oof, classes) where yi/oof are contiguous class indices.
    model_fn(seed) -> unfitted estimator; defaults to the fixed XGBoost used
    throughout, so passing nothing reproduces the main analysis exactly. Any
    estimator is fitted inside the fold, so a scaling pipeline stays leakage-free.
    """
    classes = np.unique(y)
    remap = {c: i for i, c in enumerate(classes)}
    yi = np.array([remap[v] for v in y])
    mk = model_fn or make_model

    oof = np.full(len(y), -1)
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(F, yi, groups):
        assert not set(groups[tr]) & set(groups[te]), "subject leaked across folds"
        oof[te] = mk(seed).fit(F[tr], yi[tr]).predict(F[te])
    assert (oof >= 0).all(), "some windows never landed in a test fold"
    return yi, oof, classes


def cv_oof_proba(F, y, groups, seed: int = SEED, n_folds: int = N_FOLDS):
    """Same folds and model as cv_oof, but returns out-of-fold class probabilities.

    An alarm is triggered on a probability, not on argmax, so calibration and the
    missed-alarm / false-alarm trade-off both need the probabilities rather than
    the hard labels cv_oof returns. Splits and seed match cv_oof, so argmax of
    this equals cv_oof's predictions.
    """
    classes = np.unique(y)
    remap = {c: i for i, c in enumerate(classes)}
    yi = np.array([remap[v] for v in y])

    proba = np.zeros((len(y), len(classes)), dtype=np.float64)
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(F, yi, groups):
        assert not set(groups[tr]) & set(groups[te]), "subject leaked across folds"
        proba[te] = make_model(seed).fit(F[tr], yi[tr]).predict_proba(F[te])
    return yi, proba, classes


def per_subject_macro_f1(yi, oof, subj) -> tuple[np.ndarray, np.ndarray]:
    """Macro-F1 computed within each subject.

    Averaged only over classes that subject actually performed, so a subject who
    never did an activity is not penalised for it.
    """
    subjects = np.unique(subj)
    scores = np.array([
        f1_score(yi[subj == s], oof[subj == s],
                 labels=np.unique(yi[subj == s]), average="macro", zero_division=0)
        for s in subjects
    ])
    return subjects, scores


def per_subject_class_f1(yi, oof, subj, class_idx: int) -> np.ndarray:
    """Per-subject F1 for one class; NaN for subjects lacking that class."""
    out = []
    for s in np.unique(subj):
        m = subj == s
        if not (yi[m] == class_idx).any():
            out.append(np.nan)
            continue
        out.append(f1_score(yi[m] == class_idx, oof[m] == class_idx, zero_division=0))
    return np.asarray(out)


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation: effect size for Wilcoxon."""
    d = a - b
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    rp, rn = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((rp - rn) / (rp + rn))


def bootstrap_ci(a, b, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI for the mean paired difference, resampling subjects."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_report(a, b, label_a: str, label_b: str) -> dict:
    """Subject-level paired comparison: Wilcoxon + effect size + bootstrap CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    stat, p = wilcoxon(a, b) if np.any(d != 0) else (np.nan, 1.0)
    lo, hi = bootstrap_ci(a, b)
    rbc = rank_biserial(a, b)
    res = {
        "n_subjects": int(len(a)),
        f"mean_{label_a}": float(a.mean()),
        f"mean_{label_b}": float(b.mean()),
        "mean_diff": float(d.mean()),
        "ci95_low": lo, "ci95_high": hi,
        "n_subjects_favouring_a": int((d > 0).sum()),
        "wilcoxon_p": float(p),
        "rank_biserial": rbc,
    }
    print(f"    {label_a:<20s} {a.mean():.4f} +/- {a.std():.4f}")
    print(f"    {label_b:<20s} {b.mean():.4f} +/- {b.std():.4f}")
    print(f"    diff {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"p={p:.2e}  rank-biserial={rbc:+.3f}  "
          f"({(d > 0).sum()}/{len(d)} subjects favour {label_a})")
    return res


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjustment across a family of comparisons."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        v = min(1.0, max(prev, (m - i) * p))
        adj[k] = v
        prev = v
    return adj
