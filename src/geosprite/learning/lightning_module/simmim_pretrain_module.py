import os.path
from typing import Any, Optional
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from einops import repeat

from .base_module import AdamWCosineOptimArgs, BaseModule

__all__ = ["SimMIMPreTrainingModule"]


class SimMIMPreTrainingModule(BaseModule):
    def __init__(self, optim_args: AdamWCosineOptimArgs, encoder: nn.Module, header: nn.Module, ):
        super().__init__(optim_args=optim_args, encoder=encoder, decoder=None, header=header)

        self.patch_size = self.encoder.patch_size
        self.l1_loss = nn.L1Loss(reduction="none")

    def training_step(self, batch: dict):
        x_recover = self(batch)

        # x.shape: b c h w
        # mask.shape: b c new_h new_w
        x, mask = batch["x"], batch["mask"]

        all_loss = self.l1_loss(x_recover, x)
        mask = repeat(mask, pattern="b c new_h new_w -> b c (new_h patch_size1) (new_w patch_size2)",
                      patch_size1=self.patch_size,
                      patch_size2=self.patch_size)

        # 0 value in x means nodata, model cannot recover it
        loss_mask = (mask == 1) & (x != 0)
        mask_loss = (all_loss * loss_mask).sum() / loss_mask.sum()
        self.log(name="train_mask_loss", value=mask_loss, on_step=True, sync_dist=True)

        loss_mask = (x != 0)
        global_loss = (all_loss * loss_mask).sum() / loss_mask.sum()
        self.log(name="train_global_loss", value=global_loss, on_step=True, sync_dist=True)

        # use mask loss for gradient descent
        return mask_loss

    def validation_step(self, batch: dict):
        x_recover = self(batch)

        # x.shape: b c h w
        # mask.shape: b c new_h new_w
        x, mask = batch["x"], batch["mask"]

        all_loss = self.l1_loss(x_recover, x)
        mask = repeat(mask, pattern="b c new_h new_w -> b c (new_h patch_size1) (new_w patch_size2)",
                      patch_size1=self.patch_size,
                      patch_size2=self.patch_size)

        # 0 value in x means nodata, model cannot recover it
        loss_mask = (mask == 1) & (x != 0)
        mask_loss = (all_loss * loss_mask).sum() / loss_mask.sum()
        self.log(name="val_mask_loss", value=mask_loss, on_epoch=True, sync_dist=True)

        loss_mask = (x != 0)
        global_loss = (all_loss * loss_mask).sum() / loss_mask.sum()
        self.log(name="val_global_loss", value=global_loss, on_epoch=True, sync_dist=True)

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> Any:
        def get_output_path(tif_path_):
            tif_basename_ = os.path.splitext(os.path.basename(tif_path_))[0]
            scene_folder_ = os.path.dirname(tif_path_)
            scene_name_ = os.path.basename(scene_folder_)
            predict_folder_ = os.path.dirname(scene_folder_)
            dataset_folder_ = os.path.dirname(predict_folder_)

            output_folder_ = os.path.join(dataset_folder_, "output", scene_name_)
            os.makedirs(output_folder_, exist_ok=True)

            origin_output_path_ = os.path.join(output_folder_, f"{tif_basename_}_origin.npy")
            mask_output_path_ = os.path.join(output_folder_, f"{tif_basename_}_mask.npy")
            recover_output_path_ = os.path.join(output_folder_, f"{tif_basename_}_recover.npy")

            return origin_output_path_, mask_output_path_, recover_output_path_

        x_recover = self(batch)
        x_origin, mask, tif_path_list = batch["x"], batch["mask"], batch["tif_path"]
        mask = repeat(mask, pattern="b c new_h new_w -> b c (new_h patch_size1) (new_w patch_size2)",
                      patch_size1=self.patch_size,
                      patch_size2=self.patch_size)
        x_masked = x_origin * mask

        for i, tif_path in enumerate(tif_path_list):
            origin_output_path, mask_output_path, recover_output_path = get_output_path(tif_path)
            np.save(origin_output_path, x_origin[i, :, :, :].cpu().numpy())
            np.save(mask_output_path, x_masked[i, :, :, :].cpu().numpy())
            np.save(recover_output_path, x_recover[i, :, :, :].cpu().numpy())

        return None

