"""Accuracy that accepts sens label."""
from torchmetrics.classification.accuracy import Accuracy


class FbAccuracy(Accuracy):
    """Accuracy where sens can be passed."""

    def update(self, preds: torch.Tensor, sens: torch.Tensor, target: torch.Tensor):
        """Update state with predictions and targets.

        See :ref:`references/modules:input types` for more information
        on input types.

        Args:
            preds: Predictions from model (probabilities, or labels)
            sens: Ground truth sensitive labels
            target: Ground truth labels
        """
        correct, total = _accuracy_update(
            preds,
            target,
            threshold=self.threshold,
            top_k=self.top_k,
            subset_accuracy=self.subset_accuracy,
        )

        self.correct = correct
        self.total = total
