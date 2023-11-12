from typing import Any

from .simmim_pretrain_module import SimMIMPreTrainingModule
from .classification_module import ClassificationModule
from .base_module import AdamWCosineOptimArgs

import rasterio
import os
import torch
from torch import nn

__all__ = ["SimMIMFineTuningModule"]


class SimMIMFineTuningModule(ClassificationModule):
    def __init__(self, pretrain_ckpt_path: str, optim_args: AdamWCosineOptimArgs,
                 decoder: nn.Module = None, header: nn.Module = None):
        pretrain_module = SimMIMPreTrainingModule.load_from_checkpoint(pretrain_ckpt_path)
        super().__init__(optim_args=optim_args, encoder=pretrain_module.encoder, decoder=decoder, header=header)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        def get_output_path(tif_path_):
            tif_basename_ = os.path.splitext(os.path.basename(tif_path_))[0]
            scene_folder_ = os.path.dirname(tif_path_)
            scene_name_ = os.path.basename(scene_folder_)
            sat_folder_ = os.path.dirname(scene_folder_)
            predict_folder_ = os.path.dirname(sat_folder_)

            output_folder_ = os.path.join(predict_folder_, "output", scene_name_)
            os.makedirs(output_folder_, exist_ok=True)

            output_path_ = os.path.join(output_folder_, f"{tif_basename_}.tif")

            return output_path_

        y_hat = self(batch)
        y_hat = torch.argmax(y_hat, dim=1)
        y_hat = y_hat.cpu().numpy()
        for i, tif_path in enumerate(batch["tif_path"]):
            with rasterio.open(tif_path) as src:
                mask = src.read_masks(1)
                profile = src.profile
                profile.update(dtype=rasterio.ubyte, count=1)
            output_path = get_output_path(tif_path)
            with rasterio.open(output_path, "w", **profile) as dst:
                data = y_hat[i]
                dst.write(data, 1)
                dst.write_mask(mask)


