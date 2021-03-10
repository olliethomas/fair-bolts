"""Test DataModules."""
import pytest
import torch

from fair_bolts.datamodules.adult_datamodule import AdultDataModule
from fair_bolts.datamodules.celeba_datamodule import CelebaDataModule
from fair_bolts.datamodules.cmnist_datamodule import CmnistDataModule
from fair_bolts.datamodules.compas_datamodule import CompasDataModule


def _create_dm(dm_cls):
    dm = dm_cls(batch_size=2)
    dm.prepare_data()
    dm.setup()
    return dm


@pytest.mark.parametrize(
    "dm_cls", [AdultDataModule, CompasDataModule, CelebaDataModule, CmnistDataModule]
)
def test_data_modules(dm_cls):
    """Test the datamodules."""
    dm = _create_dm(dm_cls)
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch.x.size() == torch.Size([2, *dm.size()])
    assert dm.num_classes
    assert dm.num_sens
