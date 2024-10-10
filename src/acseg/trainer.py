import argparse
from typing import Dict, List

from acseg.datasets.constants import DatasetInformation
from acseg.datasets.segmentation import get_dataset_information
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torchmetrics.classification import MulticlassJaccardIndex

DECODERS = {"fpn": smp.FPN}


class SegmentationTrainer(pl.LightningModule):
    def build_model(self) -> nn.Module:
        hparams = self.hparams["hparams"]
        decoder = DECODERS[hparams.decoder](
            encoder_name=hparams.encoder,
            encoder_weights="imagenet" if hparams.imagenet else None,
            classes=self._ds_info.n_classes,
        )
        return decoder

    def configure_optimizers(self) -> torch.optim.Optimizer:
        hparams = self.hparams["hparams"]
        optimiser = torch.optim.SGD(
            self._model.parameters(),
            lr=hparams.lr,
            weight_decay=hparams.decay,
            momentum=hparams.momentum,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimiser,
            step_size=hparams.step_lr_every_n_steps,
            gamma=hparams.lr_step_factor,
        )
        return [optimiser], [scheduler]

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters()
        # Network and Training
        self._ds_info = get_dataset_information(hparams.dataset_name)
        self._model = self.build_model()
        self._loss = nn.CrossEntropyLoss()
        self._setup_evaluation_metric()
        self._setup_class_labels()
        self.validation_outputs = []
        self.test_outputs = []

    def _setup_evaluation_metric(self):
        self._eval_metric = MulticlassJaccardIndex(
            self._ds_info.n_classes,
            average=None,
            ignore_index=self._ds_info.ignore_index,
        )

    def _setup_class_labels(self):
        self.class_labels = self._get_class_labels()

    def _get_class_labels(self):
        if self._ds_info.class_labels is None:
            class_labels = range(0, self._ds_info.n_classes)
        else:
            class_labels = self._get_class_names()
        return class_labels

    def _get_class_names(self) -> List[str]:
        class_labels = sorted(
            self._ds_info.class_labels,
            key=lambda x: x.train_id,
        )
        return [class_info.name for class_info in class_labels]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> Dict:
        images, targets = batch
        targets = targets.squeeze(1)

        predictions = self.forward(images)

        loss = self._loss(predictions, targets)
        batch_ious = self._eval_metric(predictions, targets)
        self.log("loss/train", loss, sync_dist=True)
        self.log_iou("train", batch_ious)
        return {"loss": loss}

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        images, targets = batch
        predictions = self.forward(images)
        targets = targets.squeeze(1)
        self.validation_outputs.append(
            {
                "val_loss": self._loss(predictions, targets),
                "val_iou": self._eval_metric(predictions, targets),
            }
        )

    def on_validation_epoch_end(self):
        outputs = self.validation_outputs
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_ious = torch.stack([x["val_iou"] for x in outputs]).mean(dim=0)
        self.log("loss/val", avg_loss, sync_dist=True)
        self.log_iou("val", avg_ious)
        self.validation_outputs.clear()

    def log_iou(self, stage: str, label_ious: torch.Tensor):
        for iou, label in zip(label_ious, self.class_labels):
            self.log(f"iou/{stage}_{label}", iou, sync_dist=True)
        self.log(f"iou/{stage}_mIoU", label_ious.mean(), sync_dist=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        images, targets = batch
        predictions = self.forward(images)
        targets = targets.squeeze()
        self.test_outputs.append({"test_iou": self._eval_metric(predictions, targets)})

    def on_test_epoch_end(self, outputs: List[Dict]) -> Dict:
        outputs = self.test_outputs
        avg_ious = torch.stack([x["test_iou"] for x in outputs]).mean(dim=0)
        self.log_iou("test", avg_ious)
