"""Common components for an EthicML vision datamodule."""
import os
from typing import Optional

from ethicml import Dataset, implements
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class BaseDm(LightningDataModule):
    """Base DataModule for this project."""

    def __init__(self, data_dir, batch_size, num_workers, val_split, shrink_pcnt, y_dim, s_dim):
        super().__init__()
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_pcnt = val_split
        self.shrink_pcnt = shrink_pcnt
        self.y_dim = y_dim
        self.s_dim = s_dim

        self.train_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None
        self.val_data: Optional[Dataset] = None

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
    def train_dataloader(self, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return self.make_dataloader(self.train_data, shuffle=True)

    @implements(LightningDataModule)
    def val_dataloader(self, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return self.make_dataloader(self.val_data)

    @implements(LightningDataModule)
    def test_dataloader(self, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        return self.make_dataloader(self.test_data)
