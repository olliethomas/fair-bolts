"""Adult Income Dataset."""
from typing import Union

import ethicml as em

from fair_bolts.datamodules.tabular_datamodule import TabularDataModule


class AdultDataModule(TabularDataModule):
    """UCI Adult Income Dataset."""

    def __init__(
        self, val_split: Union[float, int] = 0.2, num_workers: int = 0, batch_size: int = 32
    ):
        super().__init__(val_split=val_split, num_workers=num_workers, batch_size=batch_size)
        self.em_dataset = em.adult(split="Sex", binarize_nationality=True)
        print(self.em_dataset.class_labels)
        self.num_classes = 2
        self.num_sens = 2
