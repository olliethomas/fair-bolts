"""Accuracy per sensitive group."""

import torch
from pytorch_lightning.metrics import Accuracy


class AccuracyPerSens(Accuracy):
    """Accuracy Metric."""

    def __init__(self, sens: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sens = sens

    @property
    def __name__(self):
        return f"Accuracy_s{self.sens}"

    def update(self, preds: torch.Tensor, sens: torch.Tensor, target: torch.Tensor):
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            sens: Ground truth sensitive labels
            target: Ground truth values
        """
        mask = sens == self.sens
        if mask.sum() > 0:
            super().update(preds[mask], target[mask])
