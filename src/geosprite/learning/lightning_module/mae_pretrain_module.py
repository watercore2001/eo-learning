from pytorch_lightning import LightningModule
from transformers import ViTMAEConfig, ViTMAEForPreTraining
import timm.optim.optim_factory as optim_factory
import torch


class MaePreTrainingModule(LightningModule):

    def __init__(self, config: ViTMAEConfig):
        super().__init__()

        self.model = ViTMAEForPreTraining(config)
        self.save_hyperparameters(logger=False)

    def training_step(self, batch):
        x, _ = batch
        output = self.model(x)
        loss = output.loss
        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        # todo: how to design and config optimizer
        batch_size = 2
        lr = 1.5e-4 * batch_size / 256.
        param_groups = optim_factory.param_groups_weight_decay(self.model, 0.05)
        return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
