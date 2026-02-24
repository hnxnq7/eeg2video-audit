"""
preprocessing.py
----------------
EEG loading, epoching, normalization, and run-position metadata extraction
for the SEED-DV dataset.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

N_CHANNELS = 62
SFREQ = 200          # Hz — update if SEED-DV uses a different sampling rate
EPOCH_DURATION = 2.0 # seconds
RUN_LENGTH = 5       # clips per same-concept run


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_subject_block(
    data_dir: str,
    subject_id: int,
    block_id: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load preprocessed EEG epochs and labels for one subject/block.

    Returns
    -------
    eeg : np.ndarray, shape (n_trials, n_channels, n_times)
    labels : pd.DataFrame with columns:
        trial_idx, concept_fine, concept_coarse, color, motion,
        run_id, run_position (1–5)
    """
    sub_str = f"sub{subject_id:02d}"
    block_path = Path(data_dir) / "raw" / sub_str / f"block{block_id}.mat"

    if not block_path.exists():
        raise FileNotFoundError(f"Expected data at: {block_path}")

    # TODO: replace with actual .mat loading once data format is confirmed
    # e.g., from scipy.io import loadmat; data = loadmat(block_path)
    raise NotImplementedError(
        "Implement .mat loading once SEED-DV file format is confirmed. "
        "See data/README.md for expected structure."
    )


def load_all_subjects(
    data_dir: str,
    subject_ids: Optional[list] = None,
    block_ids: Optional[list] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load and concatenate EEG + labels across subjects and blocks.

    Returns
    -------
    eeg   : np.ndarray, shape (N, n_channels, n_times)
    labels: pd.DataFrame with subject_id and block_id columns appended
    """
    subject_ids = subject_ids or list(range(1, 21))
    block_ids   = block_ids   or list(range(1, 8))

    all_eeg, all_labels = [], []
    for sid in subject_ids:
        for bid in block_ids:
            eeg, labels = load_subject_block(data_dir, sid, bid)
            labels["subject_id"] = sid
            labels["block_id"]   = bid
            all_eeg.append(eeg)
            all_labels.append(labels)

    return np.concatenate(all_eeg, axis=0), pd.concat(all_labels, ignore_index=True)


# ─────────────────────────────────────────────
# Run-Position Metadata
# ─────────────────────────────────────────────

def assign_run_positions(labels: pd.DataFrame) -> pd.DataFrame:
    """
    Given a labels DataFrame (ordered by presentation), compute
    `run_id` and `run_position` (1–5) for every trial.

    Assumes trials are ordered sequentially and that every group of
    RUN_LENGTH consecutive trials shares the same concept.
    """
    labels = labels.copy()
    n = len(labels)
    labels["run_id"]       = np.arange(n) // RUN_LENGTH
    labels["run_position"] = (np.arange(n) % RUN_LENGTH) + 1
    return labels


# ─────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────

def normalize_epochs(
    eeg: np.ndarray,
    method: str = "zscore",
) -> np.ndarray:
    """
    Normalize EEG epochs.

    Parameters
    ----------
    eeg    : (N, C, T)
    method : 'zscore' | 'minmax'

    Returns
    -------
    eeg_norm : (N, C, T)
    """
    if method == "zscore":
        mean = eeg.mean(axis=(0, 2), keepdims=True)
        std  = eeg.std(axis=(0, 2), keepdims=True) + 1e-8
        return (eeg - mean) / std
    elif method == "minmax":
        mn = eeg.min(axis=2, keepdims=True)
        mx = eeg.max(axis=2, keepdims=True)
        return (eeg - mn) / (mx - mn + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# ─────────────────────────────────────────────
# Block-wise CV Split Helpers
# ─────────────────────────────────────────────

def blockwise_cv_splits(
    labels: pd.DataFrame,
    n_blocks: int = 7,
) -> list[Dict]:
    """
    Generate block-wise cross-validation splits as used in the original
    EEG2Video paper (leave-one-block-out).

    Returns a list of dicts with keys 'train_idx' and 'test_idx'.
    """
    splits = []
    for test_block in range(1, n_blocks + 1):
        test_mask  = labels["block_id"] == test_block
        train_mask = ~test_mask
        splits.append({
            "test_block": test_block,
            "train_idx":  np.where(train_mask)[0],
            "test_idx":   np.where(test_mask)[0],
        })
    return splits


def original_paper_split(
    labels: pd.DataFrame,
    train_blocks: list = None,
    test_block: int = 7,
) -> Dict:
    """
    Replicate the original paper's fixed split:
    train on blocks 1–6, test on block 7.
    """
    train_blocks = train_blocks or [1, 2, 3, 4, 5, 6]
    train_mask = labels["block_id"].isin(train_blocks)
    test_mask  = labels["block_id"] == test_block
    return {
        "train_idx": np.where(train_mask)[0],
        "test_idx":  np.where(test_mask)[0],
    }
