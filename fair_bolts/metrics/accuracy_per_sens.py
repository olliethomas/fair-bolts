"""Accuracy per sensitive group."""

import torch
from pytorch_lightning.metrics import Accuracy

from fair_bolts.metrics.classfication import _input_format_classification


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

        See :ref:`references/modules:input types` for more information
        on input types.

        Args:
            preds: Predictions from model (probabilities, or labels)
            sens: Ground truth sensitive labels
            target: Ground truth labels
        """
        preds, target = _input_format_classification(preds, target, self.threshold)
        assert preds.shape == target.shape

        self.correct += torch.sum(preds[sens == self.sens] == target[sens == self.sens])
        self.total += target.numel()
