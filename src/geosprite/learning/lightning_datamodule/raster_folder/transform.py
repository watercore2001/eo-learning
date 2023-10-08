# Copyright (c) GeoSprite. All rights reserved.
#
# Author: Jia Song
#
import math
import numpy as np
import torch
import warnings
from torchvision import transforms
from torchvision.transforms import functional
from torch.utils.data import DataLoader


class ToTensor:
    """ convert numpy to tensor as specified data type """
    def __init__(self, type_str: str = "torch.LongTensor"):
        self.type_str = type_str

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        # these numpy types are not supported in torch
        if data.dtype == np.uint16:
            data = data.astype(np.int32)
        if data.dtype in (np.uint32, np.uint64):
            data = data.astype(np.int64)
        data = torch.as_tensor(data)
        # can only use str convert type in tensor.type() method
        return data.type(dtype=self.type_str)


class ToNormalizedTensor:
    def __init__(self, mean: list[float] = None, std: list[float] = None):
        self.mean = mean
        self.std = std

    def init_mean_std(self, dataset) -> None:
        """
        https://towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c
        """
        channels_sum, channels_squared_sum, nums = torch.zeros((1,)), torch.zeros((1,)), torch.zeros((1,))

        if len(dataset) == 0:
            raise RuntimeError(f'dataset is empty.')

        transform_backup = dataset.image_transform, dataset.gt_transform

        dataset.image_transform = ToTensor(type_str="torch.FloatTensor")
        dataset.gt_transform = ToTensor(type_str="torch.FloatTensor")

        for x, _ in DataLoader(dataset):
            # Mean over batch, height and width
            # cannot use += when there is broadcasting
            channels_sum = channels_sum + torch.mean(x, dim=[0, 2, 3])
            channels_squared_sum = channels_squared_sum + torch.mean(x ** 2, dim=[0, 2, 3])
            nums += 1

        mean = channels_sum / nums

        # std = sqrt(E[X^2] - (E[X])^2)
        std = (channels_squared_sum / nums - mean ** 2) ** 0.5

        self.mean = [round(i, 4) for i in mean.tolist()]
        self.std = [round(i, 4) for i in std.tolist()]

        dataset.image_transform, dataset.gt_transform = transform_backup

    def __call__(self, data: np.ndarray) -> torch.Tensor:

        if isinstance(self.mean, list) and isinstance(self.std, list):
            transform_instance = transforms.Compose([ToTensor(type_str="torch.FloatTensor"),
                                                    transforms.Normalize(mean=self.mean, std=self.std)])
            return transform_instance(data)
        else:
            warnings.warn('Parameters mean and std are None. '
                          'ToRangeOneTensor is performed instead of ToNormalizedTensor')
            data = functional.to_tensor(data)
            data = data.div(np.iinfo(np.int32).max)
            return data


class ToGridTensor:

    def __init__(self, image_size: tuple[int, int], input_size: tuple[int, int],
                 padding_value: int = 0):
        self.rows = math.ceil(image_size[0] / input_size[1])
        self.cols = math.ceil(image_size[1] / input_size[1])

        self.bottom_paddings = self.rows * input_size[0] - image_size[0]
        self.right_paddings = self.cols * input_size[1] - image_size[1]

        self.padding_value = padding_value

    def __call__(self, data) -> torch.Tensor:

        if self.bottom_paddings > 0 or self.right_paddings > 0:
            data = torch.nn.functional.pad(input=data, pad=(0, self.right_paddings, 0, self.bottom_paddings),
                                           mode='constant', value=self.padding_value)

        input_grids = [n for m in torch.chunk(data, self.rows, dim=-2) for n in torch.chunk(m, self.cols, dim=-1)]

        return torch.stack(input_grids)

    def reverse_image(self, input_grids: torch.Tensor) -> torch.Tensor:
        # (grids, c, h, w) -> (grid_rows, grid_cols, c, h, w)
        # important: row is in front of col, because first chunk row
        input_grids = input_grids.reshape(self.rows, self.cols, *input_grids.shape[1:])
        image = torch.cat(list(input_grids), dim=-2, out=None)
        image = torch.cat(list(image), dim=-1, out=None)

        if self.bottom_paddings > 0 or self.right_paddings > 0:
            image = image[..., :-self.bottom_paddings, :-self.right_paddings]

        return image
