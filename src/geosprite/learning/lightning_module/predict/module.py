# Copyright (c) GeoSprite. All rights reserved.
#
# Author: Jia Song
#

from typing import Any

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor

__all__ = ['PredictModule']


class PredictModule(LightningModule):

    def __init__(self, model: nn.Module):
        super(PredictModule, self).__init__()
        self.model = model

    def forward(self, x: Tensor) -> Any:
        return self.model(x)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        # batch: Tensor (N, C, H, W)
        # self(batch): (N, logistic, H, W)
        return self(batch).argmax(1).to(torch.uint8)  # Tensor (N, H, W)
