"""Accuracy per sensitive group."""
from typing import Any, Callable, Dict, Optional

import torch
from torchmetrics.functional.classification.accuracy import _accuracy_compute, _accuracy_update
from torchmetrics.metric import Metric


class AccuracyPerSens(Metric):
    r"""Accuracy.

    Computes `Accuracy <https://en.wikipedia.org/wiki/Accuracy_and_precision>`__:
    .. math::
        \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y}_i)
    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
    tensor of predictions.
    For multi-class and multi-dimensional multi-class data with probability predictions, the
    parameter ``top_k`` generalizes this metric to a Top-K accuracy metric: for each sample the
    top-K highest probability items are considered to find the correct label.
    For multi-label and multi-dimensional multi-class inputs, this metric computes the "global"
    accuracy by default, which counts all labels or sub-samples separately. This can be
    changed to subset accuracy (which requires all labels or sub-samples in the sample to
    be correctly predicted) by setting ``subset_accuracy=True``.
    Accepts all input types listed in :ref:`references/modules:input types`.

    Args:
        num_sens:
            The number of sensitive attribute values. Makes an assumption that these labels are
            0-indexed and fits the form `range(num_sens)`.
        threshold:
            Threshold probability value for transforming probability predictions to binary
            (0,1) predictions, in the case of binary or multi-label inputs.
        top_k:
            Number of highest probability predictions considered to find the correct label, relevant
            only for (multi-dimensional) multi-class inputs with probability predictions. The
            default value (``None``) will be interpreted as 1 for these inputs.
            Should be left at default (``None``) for all other types of inputs.
        subset_accuracy:
            Whether to compute subset accuracy for multi-label and multi-dimensional
            multi-class inputs (has no effect for other input types).
            - For multi-label inputs, if the parameter is set to ``True``, then all labels for
              each sample must be correctly predicted for the sample to count as correct. If it
              is set to ``False``, then all labels are counted separately - this is equivalent to
              flattening inputs beforehand (i.e. ``preds = preds.flatten()`` and same for ``target``).
            - For multi-dimensional multi-class inputs, if the parameter is set to ``True``, then all
              sub-sample (on the extra axis) must be correct for the sample to be counted as correct.
              If it is set to ``False``, then all sub-samples are counter separately - this is equivalent,
              in the case of label predictions, to flattening the inputs beforehand (i.e.
              ``preds = preds.flatten()`` and same for ``target``). Note that the ``top_k`` parameter
              still applies in both cases, if set.
        compute_on_step:
            Forward only calls ``update()`` and return ``None`` if this is set to ``False``.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step
        process_group:
            Specify the process group on which synchronization is called.
            default: ``None`` (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather
    Example:
        >>> from torchmetrics import Accuracy
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> preds = torch.tensor([0, 2, 1, 3])
        >>> accuracy = Accuracy()
        >>> accuracy(preds, target)
        tensor(0.5000)
        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]])
        >>> accuracy = Accuracy(top_k=2)
        >>> accuracy(preds, target)
        tensor(0.6667)
    """

    def __init__(
        self,
        num_sens: int,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
        subset_accuracy: bool = False,
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

        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            raise ValueError(f"The `top_k` should be an integer larger than 0, got {top_k}")

        self.threshold = threshold
        self.top_k = top_k
        self.subset_accuracy = subset_accuracy

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
            correct, total = _accuracy_update(
                preds[sens == _s],
                target[sens == _s],
                threshold=self.threshold,
                top_k=self.top_k,
                subset_accuracy=self.subset_accuracy,
            )

            setattr(self, f"correct_{_s}", self.__getattribute__(f"correct_{_s}") + correct)
            setattr(self, f"total_{_s}", self.__getattribute__(f"total_{_s}") + total)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Computes accuracy based on inputs passed in to ``update`` previously."""
        return {
            f"sens group {_s}": _accuracy_compute(
                self.__getattribute__(f"correct_{_s}"), self.__getattribute__(f"total_{_s}")
            )
            for _s in range(self.num_sens)
        }
