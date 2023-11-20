from typing import Any

from .simmim_pretrain_module import SimMIMPreTrainingModule
from .classification_module import ClassificationModule
from .base_module import AdamWCosineOptimArgs

import torch
from torch import nn

__all__ = ["SimMIMFineTuningModule"]


class SimMIMFineTuningModule(ClassificationModule):
    def __init__(self, pretrain_ckpt_path: str, optim_args: AdamWCosineOptimArgs,
                 decoder: nn.Module = None, header: nn.Module = None):
        pretrain_module = SimMIMPreTrainingModule.load_from_checkpoint(pretrain_ckpt_path)
        super().__init__(optim_args=optim_args, encoder=pretrain_module.encoder, decoder=decoder, header=header)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        y_hat = self(batch)
        y_hat = torch.argmax(y_hat, dim=1).to(torch.uint8)
        return y_hat



