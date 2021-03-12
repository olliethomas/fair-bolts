"""Accuracy that accepts sens label."""
import torch
from pytorch_lightning.metrics import Accuracy

from fair_bolts.metrics.classfication import _input_format_classification


class FbAccuracy(Accuracy):
    """Accuracy where sens can be passed."""

    @property
    def __name__(self):
        return f"Accuracy"

    def update(self, preds: torch.Tensor, sens: torch.Tensor, target: torch.Tensor):
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            sens: Ground truth sensitive labels
            target: Ground truth values
        """
        preds, target = _input_format_classification(preds, target, self.threshold)
        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()
