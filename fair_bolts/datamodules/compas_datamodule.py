"""COMPAS Dataset."""
from typing import Union

import ethicml as em

from fair_bolts.datamodules.tabular_datamodule import TabularDataModule


class CompasDataModule(TabularDataModule):
    """COMPAS Dataset."""

    def __init__(
        self, val_split: Union[float, int] = 0.2, num_workers: int = 0, batch_size: int = 32
    ):
        super().__init__(val_split=val_split, num_workers=num_workers, batch_size=batch_size)
        self.em_dataset = em.compas(split="Sex")
