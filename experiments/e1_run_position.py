"""
e1_run_position.py
------------------
RQ1: Does benchmark decoding performance vary systematically with run position
(clip 1–5 within a same-concept run)?

Usage
-----
# Reproduce baseline block-wise CV
python experiments/e1_run_position.py --reproduce_baseline

# Run stratified position analysis
python experiments/e1_run_position.py --task concept --subject all
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
    stratified_metrics_by_run_position,
    compute_position_trend,
)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run_baseline_cv(cfg: dict, logger):
    """Reproduce block-wise 7-fold CV as in the original paper."""
    logger.info("=== Reproducing baseline block-wise CV ===")

    # TODO: load data + labels via src.preprocessing
    # TODO: for each CV fold, train model and collect predictions
    # TODO: report mean ± std accuracy across folds
    raise NotImplementedError(
        "Implement after SEED-DV data is available. "
        "See src/preprocessing.py for data loading helpers."
    )


def run_position_stratified(cfg: dict, task: str, subject_ids: list, logger):
    """
    Stratify test-set predictions by run position (1–5) to detect
    adaptation or expectation effects.
    """
    logger.info(f"=== RQ1: Run-position analysis | task={task} | subjects={subject_ids} ===")

    # TODO: load predictions from baseline CV (or re-run inference)
    # TODO: attach run_position column from labels
    # TODO: call stratified_metrics_by_run_position()

    # Placeholder — replace with real results
    position_accuracies = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}

    trend = compute_position_trend(position_accuracies)
    logger.info(f"Linear trend — slope: {trend['slope']:.4f}, R²: {trend['r_squared']:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    positions = sorted(position_accuracies.keys())
    accs = [position_accuracies[p] for p in positions]
    ax.plot(positions, accs, marker="o", linewidth=2, color="steelblue")
    ax.set_xlabel("Run Position (clip index within same-concept run)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"RQ1: Run-Position Effect — {task} decoding")
    ax.set_xticks(positions)
    ax.grid(alpha=0.3)

    slope_txt = f"slope={trend['slope']:.4f}, R²={trend['r_squared']:.3f}"
    ax.annotate(slope_txt, xy=(0.05, 0.92), xycoords="axes fraction", fontsize=9)

    save_figure(fig, f"rq1_run_position_{task}")
    plt.close(fig)

    return position_accuracies, trend


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="E1: Run-position analysis (RQ1)")
    p.add_argument("--config",              default="configs/config.yaml")
    p.add_argument("--task",                default="concept",
                   choices=["concept", "concept_coarse", "color", "motion"])
    p.add_argument("--subject",             default="all",
                   help="Subject ID or 'all'")
    p.add_argument("--reproduce_baseline",  action="store_true",
                   help="Only reproduce the baseline CV without position stratification")
    p.add_argument("--seed",                type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)
    set_seed(args.seed)
    logger = get_logger("e1_run_position")

    subject_ids = (
        list(range(1, 21))
        if args.subject == "all"
        else [int(args.subject)]
    )

    if args.reproduce_baseline:
        run_baseline_cv(cfg, logger)
    else:
        run_position_stratified(cfg, args.task, subject_ids, logger)
