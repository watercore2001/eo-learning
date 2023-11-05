from pytorch_lightning import LightningModule
import torch
from torch import nn
from einops import repeat
from torch import optim
from ..lr_scheduler import CosineAnnealingWithWarmup
import dataclasses

__all__ = ["SimMIMOptimArgs", "SimMIMPreTrainingModule"]

@dataclasses.dataclass
class SimMIMOptimArgs:
    weight_decay: float
    warmup_epochs: int
    annealing_epochs: int
    max_lr: float
    min_lr: float


class SimMIMPreTrainingModule(LightningModule):
    def __init__(self, encoder: nn.Module, header: nn.Module, optim_args: SimMIMOptimArgs):
        super().__init__()
        self.encoder = encoder
        self.header = header
        self.optim_args = optim_args

        self.loss = nn.L1Loss(reduction="none")

        # must save all hyperparameters for checkpoint
        self.save_hyperparameters(logger=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y_hat = self.encoder(x, mask)[-1]
        y_hat = self.header(y_hat)
        return y_hat

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        # x.shape: b c h w
        # mask.shape: b c new_h new_w
        x, mask = batch
        x_recovery = self(x, mask)

        all_loss = self.loss(x_recovery, x)

        mask = repeat(mask, pattern="b c new_h new_w -> b c (new_h patch_size1) (new_w patch_size2)",
                      patch_size1=self.encoder.patch_size,
                      patch_size2=self.encoder.patch_size)

        # 0 value in x means nodata, model cannot recover it
        loss_mask = (mask == 1) & (x != 0)
        loss = (all_loss * loss_mask).sum() / loss_mask.sum()
        self.log(name="train_loss", value=loss, on_step=True, sync_dist=True)
        return loss

    def get_param_groups(self):
        def check_keywords_in_name(name_: str, skip_keywords_: set[str]):
            for keyword in skip_keywords_:
                if keyword in name_:
                    return True
            return False

        skip_keywords = self.encoder.no_weight_decay_keywords()

        has_decay_param = []
        no_decay_param = []
        has_decay_name = []
        no_decay_name = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if check_keywords_in_name(name, skip_keywords):
                no_decay_param.append(param)
                no_decay_name.append(name)
            else:
                has_decay_param.append(param)
                has_decay_name.append(name)

        return [{'params': has_decay_param},
                {'params': no_decay_param, 'weight_decay': 0}]

    def configure_optimizers(self):
        params = self.get_param_groups()

        optimizer = optim.AdamW(params=params, lr=self.optim_args.max_lr, weight_decay=self.optim_args.weight_decay)
        lr_scheduler = CosineAnnealingWithWarmup(optimizer=optimizer, warmup_epochs=self.optim_args.warmup_epochs,
                                                 annealing_epochs=self.optim_args.annealing_epochs,
                                                 max_lr=self.optim_args.max_lr,
                                                 min_lr=self.optim_args.min_lr)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

