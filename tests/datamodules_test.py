"""Test DataModules."""

from fair_bolts.datamodules.adult_datamodule import AdultDataModule


def test_adult():
    """Test Adult Dataset."""
    dm = AdultDataModule()
    dm.prepare_data()
    dm.setup()
    assert dm is not None
