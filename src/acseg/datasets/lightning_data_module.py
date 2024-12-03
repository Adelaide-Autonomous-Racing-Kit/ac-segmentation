import argparse

from acseg.datasets import segmentation
from acseg.datasets.augmentation import Augmentor, ResizeNormalise
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, args: argparse.ArgumentParser):
        super().__init__()
        self.save_hyperparameters()

    def train_dataloader(self) -> DataLoader:
        hparams = self.hparams["args"]
        train_dataset = segmentation.build_dataset(
            hparams,
            transform=Augmentor(hparams),
        )
        return DataLoader(
            train_dataset,
            batch_size=hparams.batch_size,
            num_workers=hparams.n_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

    def val_dataloader(self) -> DataLoader:
        hparams = self.hparams["args"]
        val_dataset = segmentation.build_dataset(
            hparams,
            "val",
            ResizeNormalise(hparams),
        )
        return DataLoader(
            val_dataset,
            batch_size=hparams.val_batch_size,
            num_workers=hparams.n_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        hparams = self.hparams["args"]
        val_dataset = segmentation.build_dataset(
            hparams,
            "val",
            ResizeNormalise(hparams),
        )
        return DataLoader(
            val_dataset,
            batch_size=hparams.val_batch_size,
            num_workers=hparams.n_workers,
            persistent_workers=True,
        )
