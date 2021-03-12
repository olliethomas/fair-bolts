"""Accuracy per sensitive group."""
from typing import Any, Callable, Dict, Optional

import torch
from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.classification.helpers import _input_format_classification


class AccuracyPerSens(Metric):
    r"""Accuracy Metric.

    Computes `Accuracy <https://en.wikipedia.org/wiki/Accuracy_and_precision>`_:

    .. math:: \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y_i})

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
    tensor of predictions.  Works with binary, multiclass, and multilabel
    data.  Accepts logits from a model output or integer class values in
    prediction.  Works with multi-dimensional preds and target.

    Forward accepts

    - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes
    - ``target`` (long tensor): ``(N, ...)``

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument.
    This is the case for binary and multi-label logits.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    Args:
        num_sens:
            Number of sensitive labels. Assumes 0-indexed and range(num_sens) covers all values.
        threshold:
            Threshold value for binary or multi-label logits. default: 0.5
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When `None`, DDP
            will be used to perform the allgather. default: None

    Example:
        >>> from pytorch_lightning.metrics import Accuracy
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> preds = torch.tensor([0, 2, 1, 3])
        >>> accuracy = Accuracy()
        >>> accuracy(preds, target)
        tensor(0.5000)

    """

    def __init__(
        self,
        num_sens: int,
        threshold: float = 0.5,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):

        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.num_sens = num_sens

        for _s in range(self.num_sens):
            self.add_state(f"correct_{_s}", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state(f"total_{_s}", default=torch.tensor(0), dist_reduce_fx="sum")

        if not 0 < threshold < 1:
            raise ValueError(
                f"The `threshold` should be a float in the (0,1) interval, got {threshold}"
            )

        self.threshold = threshold

    def update(self, preds: torch.Tensor, sens: torch.Tensor, target: torch.Tensor):
        """Update state with predictions and targets.

        See :ref:`references/modules:input types` for more information
        on input types.

        Args:
            preds: Predictions from model (probabilities, or labels)
            sens: Ground truth sensitive labels
            target: Ground truth labels
        """
        for _s in range(self.num_sens):
            preds, target = _input_format_classification(preds, target, self.threshold)
            assert preds.shape == target.shape

            correct = torch.sum(preds[sens == _s] == target[sens == _s])
            total = target.numel()

            setattr(self, f"correct_{_s}", self.__getattribute__(f"correct_{_s}") + correct)
            setattr(self, f"total_{_s}", self.__getattribute__(f"total_{_s}") + total)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Computes accuracy based on inputs passed in to ``update`` previously."""
        return {
            f"sens group {_s}": (
                self.__getattribute__(f"correct_{_s}").float()
                / self.__getattribute__(f"total_{_s}")
            )
            for _s in range(self.num_sens)
        }
