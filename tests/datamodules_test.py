"""Test DataModules."""
import pytest
import torch

from fair_bolts.datamodules.adult_datamodule import AdultDataModule


def _create_dm(dm_cls, val_split=0.2):
    dm = dm_cls(val_split=val_split, num_workers=1, batch_size=2)
    dm.prepare_data()
    dm.setup()
    return dm


@pytest.mark.parametrize("dm_cls", [AdultDataModule])
def test_data_modules(dm_cls):
    """Test the datamodules."""
    dm = _create_dm(dm_cls)
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch.x.size() == torch.Size([2, *dm.size()])
