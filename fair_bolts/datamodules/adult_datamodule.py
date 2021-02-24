"""Adult Income Dataset."""
from typing import List, Optional, Union

import ethicml as em
import torch
from ethicml import implements
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from fair_bolts.datasets.ethicml_datasets import DataTupleDataset


class AdultDataModule(LightningDataModule):
    """UCI Adult Income Dataset."""

    def __init__(
        self,
        val_split: Union[float, int] = 0.2,
        num_workers: int = 0,
        batch_size: int = 32,
    ):
        """Adult Dataset Module.

        Args:
            val_split: Percent (float) or number (int) of samples to use for the validation split
            num_workers: How many workers to use for loading data
            batch_size: How many samples per batch to load
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = 0
        self.val_split = val_split

    def _get_splits(self, train_len: int) -> List[int]:
        """Computes split lengths for train and validation set."""
        if isinstance(self.val_split, int):
            train_len -= self.val_split
            splits = [train_len, self.val_split]
        elif isinstance(self.val_split, float):
            val_len = int(self.val_split * train_len)
            train_len -= val_len
            splits = [train_len, val_len]
        else:
            raise ValueError(f"Unsupported type {type(self.val_split)}")

        return splits

    @implements(LightningDataModule)
    def prepare_data(self):
        self.em_dataset = em.adult(split="Sex", binarize_nationality=True)
        self.dims = (
            len(self.em_dataset.discrete_features)
            + len(self.em_dataset.continuous_features),
        )

    @implements(LightningDataModule)
    def setup(self, stage: Optional[str] = None) -> None:
        self.datatuple = self.em_dataset.load(ordered=True)
        self.dataset = DataTupleDataset(
            self.datatuple,
            disc_features=self.em_dataset.discrete_features,
            cont_features=self.em_dataset.continuous_features,
        )

        num_train, num_val = self._get_splits(int(self.datatuple.x.shape[0] * 0.8))
        g_cpu = torch.Generator()
        g_cpu = g_cpu.manual_seed(self.seed)

        self.train_data, self.val_data, self.test_data = random_split(
            self.dataset,
            lengths=(
                num_train,
                num_val,
                int(self.datatuple.x.shape[0]) - num_train - num_val,
            ),
            generator=g_cpu,
        )

    def make_dataloader(self, ds, shuffle=False):
        """Make DataLoader."""
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @implements(LightningDataModule)
    def train_dataloader(
        self, shuffle: bool = False, drop_last: bool = False
    ) -> DataLoader:
        return self.make_dataloader(self.dataset, shuffle=True)

    @implements(LightningDataModule)
    def val_dataloader(
        self, shuffle: bool = False, drop_last: bool = False
    ) -> DataLoader:
        return self.make_dataloader(self.dataset)

    @implements(LightningDataModule)
    def test_dataloader(
        self, shuffle: bool = False, drop_last: bool = False
    ) -> DataLoader:
        return self.make_dataloader(self.dataset)
