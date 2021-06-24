"""Tests for models."""
import pytorch_lightning as pl

from fair_bolts.datamodules import CelebaDataModule
from fair_bolts.models.laftr_baseline import Laftr


def test_laftr(enc, adv, clf, dec) -> None:
    """Test the Laftr model."""
    dm = CelebaDataModule()
    dm.prepare_data()
    dm.setup()

    trainer = pl.Trainer(fast_dev_run=True)

    model = Laftr(
        enc=enc,
        dec=dec,
        adv=adv,
        clf=clf,
        weight_decay=1e-8,
        lr_gamma=0.999,
        disc_steps=1,
        fairness="DP",
        recon_weight=1.0,
        clf_weight=0.0,
        adv_weight=1.0,
        lr=1e-3,
    )

    trainer.fit(model, datamodule=dm)
