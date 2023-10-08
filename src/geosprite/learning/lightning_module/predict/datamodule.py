# Copyright (c) GeoSprite. All rights reserved.
#
# Author: Jia Song
#

from typing import Any, Optional, Union, List, Callable

import numpy as np
import rasterio
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from ..transforms import Transform

__all__ = ['RasterFilesModule', 'RasterArraysModule']


# class ArraySplitter:
#
#     def __init__(self, max_height, max_width):
#         self.num_heights = None
#         self.num_widths = None
#         self.max_height = max_height
#         self.max_width = max_width
#
#     def split(self, array) -> List[np.ndarray]:
#
#         if not isinstance(self.num_heights, int):
#             self.num_heights = array.shape[-2] // self.max_height + 1
#
#         if not isinstance(self.num_widths, int):
#             self.num_widths = array.shape[-1] // self.max_width + 1
#
#         results = []
#
#         for arr in np.array_split(array, self.num_heights, -2):
#             results += np.array_split(arr, self.num_widths, -1)
#
#         return results
#
#     def merge(self, arrays: List, shape: Optional[Tuple] = None) -> np.ndarray:
#
#         if not isinstance(self.num_heights, int) and isinstance(shape, tuple) and len(shape) >= 2:
#             self.num_heights = shape[-2] // self.max_height + 1
#
#         if not isinstance(self.num_widths, int) and isinstance(shape, tuple) and len(shape) >= 2:
#             self.num_widths = shape[-1] // self.max_width + 1
#
#         arr_list = [np.concatenate(arrays[i:i + self.num_widths], axis=-1)
#                     for i in range(0, self.num_heights * self.num_widths, self.num_widths)]
#
#         return np.concatenate(arr_list, axis=-2)


class RasterFiles(Dataset[Tensor]):

    def __init__(self, image_files: Union[str, List[str]], image_transform: Optional[Transform] = None) -> None:

        if image_files is None:
            image_files = []
        elif isinstance(image_files, str):
            image_files = [image_files]

        self.samples = image_files
        self.image_transform = image_transform

    def __getitem__(self, index: int) -> Tensor:
        image_file = self.samples[index]

        with rasterio.open(image_file) as dataset:
            data = dataset.read(masked=True)

            if isinstance(self.image_transform, Callable):
                data = self.image_transform(data)

            return data

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            length of the dataset
        """
        return len(self.samples)


class RasterArrays(Dataset[Any]):

    def __init__(self, image_arrays: List[np.ndarray], image_transform: Optional[Transform] = None) -> None:
        self.samples = image_arrays
        self.image_transform = image_transform

    def __getitem__(self, index: int) -> Any:
        data = self.samples[index]
        if isinstance(self.image_transform, Callable):
            data = self.image_transform(data)

        return data

    def __len__(self) -> int:
        return len(self.samples)


class RasterModule(LightningDataModule):

    def __init__(
            self,
            image_transform: Optional[Transform] = None,
            batch_size: Optional[int] = 1,
            num_workers: Optional[int] = 0,
    ):
        super().__init__()

        self.image_transform = image_transform
        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def dataset(self) -> Dataset:
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader[Any]:
        pass

    def val_dataloader(self) -> DataLoader[Any]:
        pass

    def test_dataloader(self) -> DataLoader[Any]:
        pass

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class RasterFilesModule(RasterModule):

    def __init__(
            self,
            image_files: Optional[Union[str, List[str]]] = [],
            image_transform: Optional[Transform] = None,
            batch_size: Optional[int] = 1,
            num_workers: Optional[int] = 0,
    ):
        super(RasterFilesModule, self).__init__(image_transform, batch_size, num_workers)

        self.image_files = image_files

    @property
    def dataset(self) -> Dataset:
        return RasterFiles(self.image_files, self.image_transform)


class RasterArraysModule(RasterModule):

    def __init__(
            self,
            image_arrays: Optional[List[np.ndarray]] = [],
            image_transform: Optional[Transform] = None,
            batch_size: Optional[int] = 1,
            num_workers: Optional[int] = 0,
    ):
        self.image_arrays = image_arrays

        super(RasterArraysModule, self).__init__(image_transform, batch_size, num_workers)

    @property
    def dataset(self) -> Dataset:
        return RasterArrays(self.image_arrays, self.image_transform)
