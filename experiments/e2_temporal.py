"""
e2_temporal.py
--------------
RQ2: Within the 2-second clip-aligned EEG segment, do different label types
show different early vs. late decodability profiles?

Sliding-window decoding: train/test on sub-windows of the epoch and plot
accuracy as a function of time.

Usage
-----
python experiments/e2_temporal.py --window_ms 300 --stride_ms 50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import load_config, set_seed, get_logger, save_figure
from src.evaluation import classification_metrics


SFREQ         = 200   # Hz
EPOCH_SAMPLES = 400   # 2 seconds × 200 Hz


# ─────────────────────────────────────────────
# Sliding-Window Loop
# ─────────────────────────────────────────────

def sliding_window_decode(
    eeg: np.ndarray,        # (N, C, T)
    labels: np.ndarray,     # (N,)
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    window_samples: int,
    stride_samples: int,
    model_fn,               # callable: returns a trained sklearn-style classifier
    task: str = "concept",
) -> pd.DataFrame:
    """
    For each time window, fit model on training set and evaluate on test set.

    Returns a DataFrame with columns: window_start_ms, window_center_ms, accuracy.
    """
    n_times = eeg.shape[2]
    records = []

    starts = range(0, n_times - window_samples + 1, stride_samples)
    for start in starts:
        end    = start + window_samples
        window = eeg[:, :, start:end]

        X_train = window[train_idx].reshape(len(train_idx), -1)
        X_test  = window[test_idx].reshape(len(test_idx), -1)
        y_train = labels[train_idx]
        y_test  = labels[test_idx]

        clf = model_fn()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = (y_pred == y_test).mean()
        center_ms = int((start + window_samples // 2) / SFREQ * 1000)
        records.append({"window_start_ms": int(start / SFREQ * 1000),
                        "window_center_ms": center_ms,
                        "accuracy": acc})

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_temporal_curves(curves: dict, title: str = "RQ2: Temporal Decoding"):
    """
    curves : dict of task_name → DataFrame from sliding_window_decode
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"concept": "steelblue", "color": "tomato", "motion": "seagreen"}

    for task, df in curves.items():
        color = colors.get(task, "gray")
        ax.plot(df["window_center_ms"], df["accuracy"],
                label=task, color=color, linewidth=2)

    ax.axvline(0,    linestyle="--", color="black", alpha=0.4, label="Stimulus onset")
    ax.axvline(2000, linestyle="--", color="black", alpha=0.4, label="Stimulus offset")
    ax.set_xlabel("Time relative to clip onset (ms)")
    ax.set_ylabel("Decoding Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    return fig


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run_temporal_analysis(cfg: dict, window_ms: int, stride_ms: int, logger):
    logger.info(f"=== RQ2: Sliding-window temporal analysis | "
                f"window={window_ms}ms, stride={stride_ms}ms ===")

    window_samples = int(window_ms / 1000 * SFREQ)
    stride_samples = int(stride_ms / 1000 * SFREQ)

    # TODO: load EEG + labels (all tasks) via src.preprocessing
    # TODO: use a simple LDA or LogisticRegression as model_fn for speed
    # TODO: call sliding_window_decode for each task and aggregate across CV folds

    # Placeholder — remove once data loading is implemented
    logger.warning("Data not loaded — replace TODOs with real data loading.")

    # Dummy example structure (replace with real results):
    # curves = {
    #     "concept": sliding_window_decode(...),
    #     "color":   sliding_window_decode(...),
    #     "motion":  sliding_window_decode(...),
    # }

    # fig = plot_temporal_curves(curves)
    # save_figure(fig, f"rq2_temporal_w{window_ms}_s{stride_ms}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="E2: Temporal sliding-window analysis (RQ2)")
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--window_ms",  type=int, default=300,
                   help="Sliding window size in ms")
    p.add_argument("--stride_ms",  type=int, default=50,
                   help="Stride between windows in ms")
    p.add_argument("--tasks",      nargs="+",
                   default=["concept", "color", "motion"],
                   help="Decoding tasks to analyze")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    cfg    = load_config(args.config)
    set_seed(args.seed)
    logger = get_logger("e2_temporal")

    run_temporal_analysis(cfg, args.window_ms, args.stride_ms, logger)
