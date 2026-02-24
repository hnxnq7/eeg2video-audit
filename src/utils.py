"""
utils.py
--------
Shared helpers: config loading, reproducibility, logging, plotting.
"""

import os
import random
import yaml
import logging
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def get_logger(name: str, log_dir: str = "results") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    if not logger.handlers:
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


# ─────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────

def save_figure(fig, name: str, results_dir: str = "results/figures"):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    out = Path(results_dir) / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {out}")
    return out
