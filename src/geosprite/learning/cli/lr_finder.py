from geosprite.learning.lightning_module import ClassificationModule
from geosprite.learning.model.video.encoder import TemporalSpaceEncoder
from geosprite.learning.model.time_series.encoder import Transformer, TransformerArgs
from geosprite.learning.model.image.encoder import SwinTransformerStages, SwinTransformerStagesArgs
from geosprite.learning.model.image.decoder import ReshapeDecoder
from geosprite.learning.model.image.header import ReshapeHeader

from geosprite.learning.lightning_datamodule.pastis.datamodule import PastisDataModule, DatasetArgs, DataloaderArgs

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.tuner import Tuner
from torch import nn
from torch import optim
import torch


def init_model():
    temporal_encoder = Transformer(TransformerArgs(
        embedding_dim=128, depth=4, heads=4, head_dim=32, mlp_ratio=2, dropout=0
    ))
    space_encoder = SwinTransformerStages(SwinTransformerStagesArgs(use_absolute_position_embedding=True,
                                                                    use_relative_position_embedding=False,
                                                                    embedding_dim=128, depth_in_stages=[4],
                                                                    heads_in_stages=[4], out_indices=[0],
                                                                    window_size=16,
                                                                    mlp_ratio=2, dropout=0
                                                                    ))
    ts_encoder = TemporalSpaceEncoder(image_channels=10, patch_size=2, embedding_dim=128,
                                      num_classes=20,
                                      temporal_encoder=temporal_encoder,
                                      space_encoder=space_encoder)
    decoder = ReshapeDecoder(dims_in_stages=[128, 256], out_dim=64)
    header = ReshapeHeader(num_classes=20, embedding_dim=128, patch_size=2)

    return ClassificationModule(encoder=ts_encoder, decoder=None, header=header)


def init_datamodule():
    return PastisDataModule(folds_split=[[1, 2, 3], [4], [5]],
                            dataset_args=DatasetArgs(folder="/mnt/disk/xials/dataset/PASTIS-R",
                                                     task="semantic", use_norm=True, cache=False, mem16=False,
                                                     sats=("S2",)), dataloader_args=DataloaderArgs(
            batch_size=2, num_workers=8, pin_memory=True))


class LitModel(LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = init_model()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=19)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_index: int):
        # x: (b, c, h, w)
        # y: (b, h, w)
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_index: int):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss


def main():
    torch.set_float32_matmul_precision('medium')
    model = LitModel()
    data_module = init_datamodule()
    trainer = Trainer(accelerator="gpu", default_root_dir="/home/xials/project/workspace/")
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model=model, datamodule=data_module, min_lr=1e-10, max_lr=0.1, num_training=200)
    print("best_learning:", lr_finder.suggestion())
    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.savefig("/home/xials/lr2_1.png")


if __name__ == "__main__":
    main()
