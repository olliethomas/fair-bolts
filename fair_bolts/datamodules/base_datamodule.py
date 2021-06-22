"""Common to all datamodules."""
from __future__ import annotations

from abc import abstractmethod

import pytorch_lightning as pl
from kit import implements
from torch.utils.data import DataLoader, Dataset


class BaseDataModule(pl.LightningDataModule):
    """Base DataModule of both Tabular and Vision DataModules."""

    def __init__(
        self,
        batch_size: int,
        val_split: float | int,
        test_split: float | int,
        num_workers: int,
        seed: int,
        persist_workers: bool,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.persist_workers = persist_workers

    @staticmethod
    def _get_splits(train_len: int, val_split: int | float) -> list[int]:
        """Computes split lengths for train and validation set."""
        if isinstance(val_split, int):
            train_len -= val_split
            splits = [train_len, val_split]
        elif isinstance(val_split, float):
            val_len = int(val_split * train_len)
            train_len -= val_len
            splits = [train_len, val_len]
        else:
            raise ValueError(f"Unsupported type {type(val_split)}")

        return splits

    def make_dataloader(
        self, ds: Dataset, shuffle: bool = False, drop_last: bool = False
    ) -> DataLoader:
        """Make DataLoader."""
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=drop_last,
            persistent_workers=self.persist_workers,
        )

    @implements(pl.LightningDataModule)
    def train_dataloader(self, shuffle: bool = False, drop_last: bool = True) -> DataLoader:
        return self.make_dataloader(self.train_data, shuffle=True, drop_last=drop_last)

    @implements(pl.LightningDataModule)
    def val_dataloader(self, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return self.make_dataloader(self.val_data)

    @implements(pl.LightningDataModule)
    def test_dataloader(self, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return self.make_dataloader(self.test_data)

    @property
    @abstractmethod
    def train_data(self) -> Dataset:
        ...

    @property
    @abstractmethod
    def val_data(self) -> Dataset:
        ...

    @property
    @abstractmethod
    def test_data(self) -> Dataset:
        ...
