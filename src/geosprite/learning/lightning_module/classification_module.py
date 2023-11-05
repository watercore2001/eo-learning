import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from .metrics import generate_classification_metric, separate_classes_metric
import os

__all__ = ["ClassificationModule"]


class ClassificationModule(LightningModule):
    def __init__(self, encoder: nn.Module = None, decoder: nn.Module = None, header: nn.Module = None,
                 ignore_index: int = None):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.header = header

        # important: ignore_index is the pixel value corresponding to the ignored class
        # can not use -1 to ignore the last class, -100 is the default value
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=ignore_index or -100)

        # assert hasattr(header, "num_classes"), f"decoder {header.__class__} doesn't hava num_classes attribute"
        num_classes = header.num_classes
        # metrics
        global_metrics, classes_metrics, confusion_matrix = generate_classification_metric(num_classes=num_classes,
                                                                                           ignore_index=ignore_index)

        self.val_global_metric = global_metrics.clone(prefix="val_")
        self.test_global_metric = global_metrics.clone(prefix="test_")
        self.confusion_matrix = confusion_matrix

        self.val_classes_metric = classes_metrics.clone(prefix="val_")
        self.test_classes_metric = classes_metrics.clone(prefix="test_")

        # must save all hyperparameters for checkpoint
        self.save_hyperparameters(logger=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.encoder(x)
        y_hat = self.decoder(y_hat) if self.decoder else y_hat[0]
        y_hat = self.header(y_hat) if self.header else y_hat
        return y_hat

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_index: int):
        # x: (b, c, h, w)
        # y: (b, h, w)
        x, y = batch
        y_hat = self(x)
        loss = self.cross_entropy_loss(y_hat, y)
        self.log(name="train_loss", value=loss, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_index: int):
        x, y = batch
        y_hat = self(x)
        loss = self.cross_entropy_loss(y_hat, y)
        self.log(name="val_loss", value=loss, sync_dist=True)
        self.val_global_metric.update(y_hat, y)
        self.val_classes_metric.update(y_hat, y)

    def on_validation_epoch_end(self) -> None:
        global_metric_value = self.val_global_metric.compute()
        classes_metric_value = separate_classes_metric(self.val_classes_metric.compute())

        metric_values = {**global_metric_value, **classes_metric_value}
        self.log_dict(metric_values, sync_dist=True)

        self.val_global_metric.reset()
        self.val_classes_metric.reset()

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_index: int):
        x, y = batch
        y_hat = self(x)
        self.test_global_metric.update(y_hat, y)
        self.test_classes_metric.update(y_hat, y)

        matrix = self.confusion_matrix(y_hat, y)
        save_path = os.path.join(self.logger.save_dir, self.logger.name, self.logger.version, "confusion_matrix.pt")
        torch.save(matrix, save_path)

    def on_test_epoch_end(self):
        global_metric_value = self.test_global_metric.compute()
        classes_metric_value = separate_classes_metric(self.test_classes_metric.compute())

        metric_values = {**global_metric_value, **classes_metric_value}
        self.log_dict(metric_values, sync_dist=True)

        self.test_global_metric.reset()
        self.test_classes_metric.reset()
