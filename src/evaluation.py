"""
evaluation.py
-------------
Metrics for classification tasks (accuracy, balanced accuracy, AUC)
and reconstruction tasks (SSIM, CLIP-space distance).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from typing import Optional


# ─────────────────────────────────────────────
# Classification Metrics
# ─────────────────────────────────────────────

def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    task: str = "concept",
) -> dict:
    """
    Compute standard classification metrics.

    Parameters
    ----------
    y_true  : ground-truth labels
    y_pred  : predicted labels
    y_score : class probabilities (needed for AUC)
    task    : 'concept' | 'color' | 'motion'

    Returns
    -------
    dict of metric name → float
    """
    metrics = {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }

    if task == "motion" and y_score is not None:
        # Binary fast/slow — use ROC-AUC
        metrics["auc"] = roc_auc_score(y_true, y_score[:, 1])

    return metrics


def stratified_metrics_by_run_position(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    run_positions: np.ndarray,
) -> dict:
    """
    Compute accuracy per run position (1–5).

    Returns
    -------
    dict mapping run_position (int) → accuracy (float)
    """
    results = {}
    for pos in sorted(np.unique(run_positions)):
        mask = run_positions == pos
        results[int(pos)] = accuracy_score(y_true[mask], y_pred[mask])
    return results


# ─────────────────────────────────────────────
# Trend Analysis (for RQ1)
# ─────────────────────────────────────────────

def compute_position_trend(position_accuracies: dict) -> dict:
    """
    Fit a linear regression over run-position accuracies to quantify
    monotonic adaptation effects.

    Returns
    -------
    dict with 'slope', 'intercept', 'r_squared'
    """
    from scipy import stats

    positions = sorted(position_accuracies.keys())
    accs = [position_accuracies[p] for p in positions]
    slope, intercept, r, _, _ = stats.linregress(positions, accs)
    return {"slope": slope, "intercept": intercept, "r_squared": r ** 2}


# ─────────────────────────────────────────────
# Reconstruction Metrics (for RQ3)
# ─────────────────────────────────────────────

def ssim_score(img_true: np.ndarray, img_pred: np.ndarray) -> float:
    """
    Structural Similarity Index between two images (H, W, C) in [0, 1].
    Requires scikit-image.
    """
    from skimage.metrics import structural_similarity as ssim
    return ssim(img_true, img_pred, channel_axis=-1, data_range=1.0)


def clip_distance(
    feat_true: np.ndarray,
    feat_pred: np.ndarray,
) -> float:
    """
    Cosine distance in CLIP embedding space.
    Lower = more similar.

    Parameters
    ----------
    feat_true, feat_pred : (D,) unit-normalized CLIP vectors
    """
    cos_sim = np.dot(feat_true, feat_pred) / (
        np.linalg.norm(feat_true) * np.linalg.norm(feat_pred) + 1e-8
    )
    return float(1.0 - cos_sim)


# ─────────────────────────────────────────────
# Split Robustness Summary (for RQ3)
# ─────────────────────────────────────────────

def split_robustness_summary(per_fold_metrics: list[dict]) -> dict:
    """
    Compute mean, std, and worst-vs-best gap across held-out blocks.

    Parameters
    ----------
    per_fold_metrics : list of metric dicts, one per held-out block

    Returns
    -------
    summary dict
    """
    keys = per_fold_metrics[0].keys()
    summary = {}
    for k in keys:
        vals = np.array([m[k] for m in per_fold_metrics])
        summary[k] = {
            "mean":          float(vals.mean()),
            "std":           float(vals.std()),
            "min":           float(vals.min()),
            "max":           float(vals.max()),
            "worst_best_gap": float(vals.max() - vals.min()),
        }
    return summary
