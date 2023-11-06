import numpy as np
import random

import torch
from einops import repeat


class MaskGenerator:
    def __init__(self, image_size: int, channels: int, mask_patch_size: int, model_patch_size: int, mask_ratio: float):
        self.image_size = image_size
        self.channels = channels
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.image_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.scale = self.mask_patch_size // self.model_patch_size
        self.mask_patch_num_one_side = self.image_size // self.mask_patch_size
        self.model_patch_num_one_side = self.image_size // self.model_patch_size

    def mask_in_space(self):
        patch_num_pre_band = self.mask_patch_num_one_side ** 2
        mask_num = int(np.ceil(patch_num_pre_band * self.mask_ratio))

        mask_idx = np.random.permutation(patch_num_pre_band)[:mask_num]

        mask = np.zeros(patch_num_pre_band, dtype=int)
        mask[mask_idx] = 1

        mask = repeat(mask, pattern="(h w) -> c (h s1) (w s2)",
                      c=self.channels,
                      h=self.mask_patch_num_one_side,
                      w=self.mask_patch_num_one_side,
                      s1=self.scale,
                      s2=self.scale)
        return torch.Tensor(mask)

    def mask_in_band(self):
        mask_band_num = int(np.ceil(self.channels * self.mask_ratio))

        mask_idx = np.random.permutation(self.channels)[:mask_band_num]

        mask = np.zeros(self.channels, dtype=int)
        mask[mask_idx] = 1

        mask = repeat(mask, pattern="c -> c h w", h=self.model_patch_num_one_side, w=self.model_patch_num_one_side)
        return torch.Tensor(mask)

    def mask_in_space_and_band(self):
        patch_num = self.channels * self.mask_patch_num_one_side ** 2
        mask_num = int(np.ceil(patch_num * self.mask_ratio))

        mask_idx = np.random.permutation(patch_num)[:mask_num]
        mask = np.zeros(patch_num, dtype=int)
        mask[mask_idx] = 1

        mask = repeat(mask, pattern="(c h w) -> c (h s1) (w s2)",
                      c=self.channels,
                      h=self.mask_patch_num_one_side,
                      w=self.mask_patch_num_one_side,
                      s1=self.scale,
                      s2=self.scale)

        return torch.Tensor(mask)

    def __call__(self):
        mask_funcs = [self.mask_in_space, self.mask_in_band, self.mask_in_space_and_band]
        func = random.choice(mask_funcs)

        return func()
