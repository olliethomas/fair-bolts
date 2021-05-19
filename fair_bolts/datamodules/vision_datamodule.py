"""Common components for an EthicML vision datamodule."""
from __future__ import annotations

import os

from ethicml import Dataset

from fair_bolts.datamodules.base_datamodule import BaseDataModule


class VisionBaseDataModule(BaseDataModule):
    """Base DataModule for this project."""

    def __init__(
        self,
        data_dir: str | None,
        batch_size: int,
        num_workers: int,
        val_split: float | int,
        test_split: float | int,
        y_dim: int,
        s_dim: int,
        seed: int,
        persist_workers: bool,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            persist_workers=persist_workers,
        )
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.y_dim = y_dim
        self.s_dim = s_dim
        self.seed = seed

        self._train_data: Dataset | None = None
        self._test_data: Dataset | None = None
        self._val_data: Dataset | None = None

    @property
    def train_data(self) -> Dataset:
        return self._train_data

    @property
    def val_data(self) -> Dataset:
        return self._val_data

    @property
    def test_data(self) -> Dataset:
        return self._test_data
