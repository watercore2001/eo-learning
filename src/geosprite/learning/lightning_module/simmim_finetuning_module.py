from .simmim_pretrain_module import SimMIMPreTrainingModule
from .classification_module import ClassificationModule
from .base_module import AdamWCosineOptimArgs

from torch import nn

__all__ = ["SimMIMFineTuningModule"]


class SimMIMFineTuningModule(ClassificationModule):
    def __init__(self, pretrain_ckpt_path: str, optim_args: AdamWCosineOptimArgs,
                 decoder: nn.Module = None, header: nn.Module = None):
        pretrain_module = SimMIMPreTrainingModule.load_from_checkpoint(pretrain_ckpt_path)
        super().__init__(optim_args=optim_args, encoder=pretrain_module.encoder, decoder=decoder, header=header)
