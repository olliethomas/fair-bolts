"""Classification helpers."""
from typing import Tuple

import torch


def _input_format_classification(
    preds: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert preds and target tensors into label tensors.

    Args:
        preds: either tensor with labels, tensor with probabilities/logits or
            multilabel tensor
        target: tensor with ground true labels
        threshold: float used for thresholding multilabel input

    Returns:
        preds: tensor with labels
        target: tensor with labels
    """
    if not (preds.ndim == target.ndim or preds.ndim == target.ndim + 1):
        raise ValueError(
            "preds and target must have same number of dimensions, or one additional dimension for preds"
        )

    if preds.ndim == target.ndim + 1:
        # multi class probabilites
        preds = torch.argmax(preds, dim=1)

    if preds.ndim == target.ndim and preds.is_floating_point():
        # binary or multilabel probablities
        preds = (preds >= threshold).long()
    return preds, target
