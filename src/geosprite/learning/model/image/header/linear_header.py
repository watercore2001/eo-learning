from torch import nn
import torch.nn.functional as f
from .base_header import BaseHeader


class LinearHeader(BaseHeader):
    def __init__(self, num_classes: int, input_dim: int, scale_factor: int):
        super().__init__(num_classes)
        # the first dimension is consist of batch size and num classes
        self.class_conv = nn.Conv2d(input_dim, num_classes, kernel_size=1)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.class_conv(x)
        x = f.interpolate(x, scale_factor=self.scale_factor, mode="bilinear")
        return x