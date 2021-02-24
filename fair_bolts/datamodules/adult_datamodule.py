"""Adult Income Dataset."""
from typing import Optional

import ethicml as em
import torch
from ethicml import implements
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from fair_bolts.datasets.ethicml_datasets import DataTupleDataset


class AdultDataModule(LightningDataModule):
    """UCI Adult Income Dataset."""

    def __init__(self):
        super().__init__()
        self.batch_size = 16
        self.num_workers = 0
        self.seed = 0

    @implements(LightningDataModule)
    def prepare_data(self):
        self.em_dataset = em.adult(split="Sex", binarize_nationality=True)

    @implements(LightningDataModule)
    def setup(self, stage: Optional[str] = None) -> None:
        self.datatuple = self.em_dataset.load(ordered=True)
        self.dataset = DataTupleDataset(
            self.datatuple,
            disc_features=self.em_dataset.discrete_features,
            cont_features=self.em_dataset.continuous_features,
        )

        num_train = int(self.datatuple.x.shape[0] * 0.8)
        num_val = int(num_train * 0.1)
        g_cpu = torch.Generator()
        g_cpu = g_cpu.manual_seed(self.seed)

        self.train_data, self.val_data, self.test_data = random_split(
            self.dataset,
            lengths=(
                num_train - num_val,
                num_val,
                int(self.datatuple.x.shape[0]) - num_train,
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
