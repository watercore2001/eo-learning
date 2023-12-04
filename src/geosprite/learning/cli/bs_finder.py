from geosprite.learning.lightning_module import SimMIMPreTrainingModule, AdamWCosineOptimArgs
from geosprite.learning.model.image.encoder import SwinTransformerV2ForSimMIMLarge
from geosprite.learning.model.image.header import ReshapeHeaderForSwinLarge

from geosprite.learning.lightning_datamodule.lucc.dataset import LuccPretrainDataset, LuccPretrainDatasetArgs
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.tuner import Tuner
from torch.utils.data import DataLoader


class LitModule(SimMIMPreTrainingModule):
    def __init__(self, batch_size: int):
        encoder = SwinTransformerV2ForSimMIMLarge(image_channels=10)
        header = ReshapeHeaderForSwinLarge(num_classes=10)
        optim_args = AdamWCosineOptimArgs(weight_decay=0.05, warmup_epochs=10, annealing_epochs=90,
                                          max_lr=1e-4, min_lr=1e-5)
        super().__init__(encoder=encoder, header=header, optim_args=optim_args)
        self.batch_size = batch_size
        dataset_args = LuccPretrainDatasetArgs(folders=["/mnt/data1/dataset/sentinel-s2-l2a/"], image_size=512,
                                               mask_patch_size=32, model_patch_size=4,
                                               mask_ratio=0.5, use_aug=True)
        self.train_dataset = LuccPretrainDataset(dataset_args)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size | self.hparams.batch_size)


def main():
    torch.set_float32_matmul_precision('medium')
    model = LitModule(batch_size=8)
    trainer = Trainer(accelerator="gpu", devices=1, default_root_dir="/home/xials/code/eo-learning/workspace/",
                      max_steps=10000000, max_epochs=100)
    tuner = Tuner(trainer)
    tuner.scale_batch_size(model=model)
    print(f"find batch size: {model.batch_size}")


