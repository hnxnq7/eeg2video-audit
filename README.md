# Vision vs. Protocol Effects in EEG2Video

**A Validity Audit of the SEED-DV Benchmark**

> Run-Position, Temporal Signatures, and Split Robustness on SEED-DV Benchmarks

Inspired by [EEG2Video](https://github.com/XuanhaoLiu/EEG2Video) | [Paper (OpenReview)](https://openreview.net/pdf?id=RfsfRn9OFd)

---

## Overview

This repository contains the full codebase for our validity audit of the SEED-DV EEG benchmark. We investigate three open evaluation questions left unaddressed by the original EEG2Video paper:

| Research Question | Focus |
|---|---|
| **RQ1** | Does decoding performance vary systematically by **run position** (clip 1–5 within a same-concept run)? |
| **RQ2** | Within the 2-second stimulus window, do labels show different **early vs. late decodability profiles**? |
| **RQ3** | How sensitive are benchmark conclusions—especially video reconstruction—to the **choice of held-out block**? |

---

## Repository Structure

```
eeg2video-audit/
├── data/                   # SEED-DV dataset (see data/README.md for download instructions)
├── src/
│   ├── preprocessing.py    # EEG loading, epoching, normalization
│   ├── models.py           # Baseline classifiers (EEGNet, linear, etc.)
│   ├── evaluation.py       # Metrics: accuracy, AUC, SSIM, CLIP distance
│   └── utils.py            # Shared helpers
├── experiments/
│   ├── e1_run_position.py  # RQ1: Run-position stratified evaluation
│   ├── e2_temporal.py      # RQ2: Sliding-window temporal decoding curves
│   └── e3_robustness.py    # RQ3: Block-rotation split robustness
├── notebooks/
│   ├── 01_baseline_reproduction.ipynb
│   ├── 02_run_position_analysis.ipynb
│   ├── 03_temporal_analysis.ipynb
│   └── 04_split_robustness.ipynb
├── results/                # Generated figures and result CSVs (git-ignored by default)
├── configs/
│   └── config.yaml         # All hyperparameters and paths
├── requirements.txt
└── README.md
```

---

## Dataset

We use the **SEED-DV** dataset:
- 62-channel EEG from 20 subjects watching 1,400 short natural videos
- 40 concept categories, 2-second clip-aligned segments
- Labels: Concept (fine/coarse), Color, Fast/Slow motion

**Download instructions:** See [`data/README.md`](data/README.md)

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/eeg2video-audit.git
cd eeg2video-audit
pip install -r requirements.txt
```

Edit `configs/config.yaml` to point to your local SEED-DV data directory.

---

## Reproducing Baseline

```bash
python experiments/e1_run_position.py --reproduce_baseline
```

This replicates the block-wise 7-fold CV classification results from the original paper before applying our stratified audit.

---

## Running Experiments

### E1 — Run-Position / Adaptation Effects (RQ1)
```bash
python experiments/e1_run_position.py --task concept --subject all
```
Outputs: Per-position accuracy plots, linear trend slopes.

### E2 — Temporal Sliding-Window Analysis (RQ2)
```bash
python experiments/e2_temporal.py --window_ms 300 --stride_ms 50
```
Outputs: Temporal decoding curves for concept, color, and fast/slow tasks.

### E3 — Split Robustness (RQ3)
```bash
python experiments/e3_robustness.py --mode rotate_all
```
Outputs: Per-fold SSIM, CLIP distance variance; worst-vs-best gap table.

---

## Timeline

| Weeks | Milestone |
|---|---|
| 1–2 | Dataset pipeline; baseline reproduction |
| 3–4 | RQ1 run-position evaluation |
| 5–6 | RQ2 temporal decoding curves |
| 7–8 | RQ3 split robustness & reconstruction variance |
| 9–10 | Results consolidation, figures, confidence intervals |
| 11–12 | (Buffer) Interpretability analysis; final report |

---

## Citation

If you use this codebase, please also cite the original EEG2Video work:
```
@inproceedings{liu2024eeg2video,
  title={EEG2Video: Towards Decoding Dynamic Visual Perception from EEG Signals},
  author={Liu, Xuanhao et al.},
  booktitle={OpenReview},
  year={2024}
}
```
