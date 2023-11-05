from geosprite.learning.lightning_module import SimMIMPreTrainingModule, SimMIMOptimArgs
from geosprite.learning.model.image.encoder import SwinTransformerForSimMIMBase
from geosprite.learning.model.image.header import ReshapeHeaderForSwinBase

from geosprite.learning.lightning_datamodule.lucc import DatasetArgs, BaseDataloaderArgs, LuccDataModule

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.tuner import Tuner

def init_model():
    encoder = SwinTransformerForSimMIMBase(image_channels=10)
    header = ReshapeHeaderForSwinBase(num_classes=10)
    optim_args = SimMIMOptimArgs(weight_decay=0.05, warmup_epochs=10, annealing_epochs=90, max_lr=1e-4, min_lr=1e-5)
    return SimMIMPreTrainingModule(encoder=encoder, header=header, optim_args=optim_args)


def init_datamodule():
    return LuccDataModule(dataset_args=DatasetArgs(folder="/mnt/data4/sentinel-s2-l2a-cogs/", image_size=512,
                                                   num_channels=10, mask_patch_size=32, model_patch_size=4,
                                                   mask_ratio=0.5, use_norm=True),
                          dataloader_args=BaseDataloaderArgs(batch_size=32, num_workers=16, pin_memory=True))


def main():
    torch.set_float32_matmul_precision('medium')
    model = init_model()
    datamodule = init_datamodule()
    trainer = Trainer(accelerator="gpu", devices=1, default_root_dir="/home/xials/code/eo-learning/workspace/")
    tuner = Tuner(trainer)
    tuner.scale_batch_size(model=model, datamodule=datamodule)
    print("find batch size: {model.batch_size}")

