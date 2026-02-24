"""
models.py
---------
Baseline EEG classifiers for SEED-DV decoding tasks.
Includes a lightweight EEGNet-style CNN and a linear baseline.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────
# EEGNet (simplified)
# ─────────────────────────────────────────────

class EEGNet(nn.Module):
    """
    Compact EEGNet implementation for EEG classification.
    Reference: Lawhern et al. (2018) https://arxiv.org/abs/1611.08024

    Parameters
    ----------
    n_channels   : number of EEG channels (default 62)
    n_times      : number of time points per epoch
    n_classes    : number of output classes
    F1           : number of temporal filters
    D            : depth multiplier
    F2           : number of pointwise filters
    dropout      : dropout rate
    """

    def __init__(
        self,
        n_channels: int = 62,
        n_times: int = 400,
        n_classes: int = 40,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.temporal_conv = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise_conv = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        self.sep_conv = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Compute flattened size dynamically
        self._flat_size = self._get_flat_size(n_channels, n_times, F1, D, F2)
        self.classifier = nn.Linear(self._flat_size, n_classes)

    def _get_flat_size(self, n_channels, n_times, F1, D, F2):
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_times)
            x = self._forward_features(x)
        return x.shape[1]

    def _forward_features(self, x):
        x = F.elu(self.bn1(self.temporal_conv(x)))
        x = F.elu(self.bn2(self.depthwise_conv(x)))
        x = self.drop1(self.pool1(x))
        x = F.elu(self.bn3(self.sep_conv(x)))
        x = self.drop2(self.pool2(x))
        return x.flatten(1)

    def forward(self, x):
        """x : (batch, 1, n_channels, n_times)"""
        x = self._forward_features(x)
        return self.classifier(x)


# ─────────────────────────────────────────────
# Linear Baseline
# ─────────────────────────────────────────────

class LinearEEGClassifier(nn.Module):
    """
    Flat linear baseline: mean-power features → linear classifier.
    Useful as a lower-bound reference.
    """

    def __init__(self, n_channels: int = 62, n_times: int = 400, n_classes: int = 40):
        super().__init__()
        self.fc = nn.Linear(n_channels * n_times, n_classes)

    def forward(self, x):
        """x : (batch, 1, n_channels, n_times)"""
        return self.fc(x.flatten(1))


# ─────────────────────────────────────────────
# Model Factory
# ─────────────────────────────────────────────

def build_model(
    model_name: str,
    n_channels: int,
    n_times: int,
    n_classes: int,
    **kwargs,
) -> nn.Module:
    """
    Instantiate a model by name.

    Parameters
    ----------
    model_name : 'eegnet' | 'linear'
    """
    name = model_name.lower()
    if name == "eegnet":
        return EEGNet(n_channels=n_channels, n_times=n_times, n_classes=n_classes, **kwargs)
    elif name == "linear":
        return LinearEEGClassifier(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from ['eegnet', 'linear'].")


# ─────────────────────────────────────────────
# Training Loop (minimal)
# ─────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Returns (accuracy, all_preds, all_labels).
    """
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(y.numpy())
    preds  = np.concatenate(preds)
    labels = np.concatenate(labels)
    acc    = (preds == labels).mean()
    return acc, preds, labels
