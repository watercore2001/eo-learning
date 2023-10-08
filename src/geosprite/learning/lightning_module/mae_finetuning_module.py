import torch
from torch import nn
from geosprite.learning.lightning_module import MaePreTrainingModule, ClassificationModule


class MaeFineTuningModule(ClassificationModule):
    def __init__(self, pretrain_ckpt_path: str, decoder: nn.Module = None, header: nn.Module = None,
                 ignore_index: int = None):
        super().__init__(decoder=decoder, header=header, ignore_index=ignore_index)

        pretrain_model = MaePreTrainingModule.load_from_checkpoint(pretrain_ckpt_path)
        pretrain_model.model.config.mask_ratio = 0

        self.encoder = pretrain_model.model.vit
        self.save_hyperparameters(logger=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.encoder(x)

        y_hat = self.decoder(y_hat) if self.decoder else y_hat[0]
        y_hat = self.header(y_hat) if self.header else y_hat
        return y_hat
