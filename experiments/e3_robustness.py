"""
e3_robustness.py
----------------
RQ3: How sensitive are benchmark conclusions—especially video reconstruction
results—to the choice of the held-out block?

Replaces the original "train blocks 1–6, test block 7" split with a full
leave-one-block-out rotation, quantifying variance in reconstruction quality.

Usage
-----
# Full block-rotation robustness test
python experiments/e3_robustness.py --mode rotate_all

# Reproduce original paper's single fixed split (block 7 held out)
python experiments/e3_robustness.py --mode original
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import load_config, set_seed, get_logger, save_figure
from src.evaluation import (
    classification_metrics,
    split_robustness_summary,
    ssim_score,
    clip_distance,
)
from src.preprocessing import blockwise_cv_splits, original_paper_split


N_BLOCKS = 7


# ─────────────────────────────────────────────
# Classification Split Robustness
# ─────────────────────────────────────────────

def evaluate_classification_robustness(cfg: dict, logger) -> pd.DataFrame:
    """
    Run leave-one-block-out CV and report per-fold + summary metrics.
    """
    logger.info("=== RQ3: Classification split robustness ===")

    # TODO: load EEG + labels via src.preprocessing
    # TODO: for each fold in blockwise_cv_splits(), train and evaluate
    # TODO: collect per-fold metrics

    per_fold = []  # list of metric dicts, one per test block
    # Example structure once implemented:
    # for split in blockwise_cv_splits(labels):
    #     ...train, predict...
    #     per_fold.append({"test_block": split["test_block"], "accuracy": acc, ...})

    if not per_fold:
        logger.warning("No fold results — implement data loading first.")
        return pd.DataFrame()

    summary = split_robustness_summary(per_fold)
    logger.info(f"Classification robustness summary: {summary}")
    return pd.DataFrame(per_fold)


# ─────────────────────────────────────────────
# Reconstruction Split Robustness
# ─────────────────────────────────────────────

def evaluate_reconstruction_robustness(cfg: dict, logger) -> pd.DataFrame:
    """
    Rotate the held-out block for the video reconstruction pipeline
    and measure SSIM and CLIP-space distance variance.

    If full video synthesis is too expensive, falls back to CLIP distance
    in latent space (see cfg['robustness']['use_clip_fallback']).
    """
    use_clip_fallback = cfg.get("robustness", {}).get("use_clip_fallback", True)
    logger.info(f"=== RQ3: Reconstruction split robustness | "
                f"clip_fallback={use_clip_fallback} ===")

    per_fold = []

    for test_block in range(1, N_BLOCKS + 1):
        logger.info(f"  Evaluating with test_block={test_block}")

        # TODO: load EEG latents / video features for this split
        # TODO: run reconstruction model (or load pre-computed CLIP features)
        # TODO: compute SSIM or CLIP distance for each test trial

        # Placeholder
        fold_metrics = {
            "test_block":   test_block,
            "ssim":         np.nan,   # replace with real SSIM
            "clip_distance": np.nan,  # replace with real CLIP distance
        }
        per_fold.append(fold_metrics)

    df = pd.DataFrame(per_fold)
    logger.info(f"\nPer-fold results:\n{df.to_string(index=False)}")

    # Summary
    numeric_cols = [c for c in df.columns if c != "test_block"]
    summary = split_robustness_summary(
        [{k: row[k] for k in numeric_cols} for _, row in df.iterrows()]
    )
    logger.info(f"\nRobustness summary: {summary}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["ssim", "clip_distance"]):
        ax.bar(df["test_block"], df[metric], color="steelblue", alpha=0.8)
        ax.set_xlabel("Held-out Block")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"RQ3: {metric} across block holdouts")
        ax.set_xticks(range(1, N_BLOCKS + 1))
        ax.axhline(df[metric].mean(), linestyle="--", color="tomato",
                   label=f"mean={df[metric].mean():.3f}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "rq3_reconstruction_robustness")
    plt.close(fig)

    return df


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="E3: Split robustness analysis (RQ3)")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--mode",   default="rotate_all",
                   choices=["original", "rotate_all"],
                   help="'original' = paper's fixed split; 'rotate_all' = full rotation")
    p.add_argument("--task",   default="reconstruction",
                   choices=["classification", "reconstruction", "both"])
    p.add_argument("--seed",   type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    cfg    = load_config(args.config)
    set_seed(args.seed)
    logger = get_logger("e3_robustness")

    if args.task in ("classification", "both"):
        evaluate_classification_robustness(cfg, logger)

    if args.task in ("reconstruction", "both"):
        evaluate_reconstruction_robustness(cfg, logger)
