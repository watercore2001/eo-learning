from torch.utils.data import Dataset
import glob
import os
import torch
import rasterio
from timm.models.layers import to_2tuple
import dataclasses
from typing import Callable
from .transform import ToGridTensor


@dataclasses.dataclass
class DatesetArgs:
    image_folder: str
    gt_folder: str
    image_transform: Callable = None
    gt_transform: Callable = None
    input_size: int | tuple[int, int] = None
    suffix_filter: str = None


class RasterFolderDataset(Dataset):
    def __init__(
            self,
            args: DatesetArgs
    ):
        self.item_ids = []
        self.item_cache = []
        self.cached_file_id = None

        self.image_transform = args.image_transform
        self.gt_transform = args.gt_transform
        self.input_size = args.input_size if args.input_size is None else to_2tuple(args.input_size)

        for file_id, image_file in enumerate(glob.iglob(
                os.path.join(args.image_folder, f'*.{args.suffix_filter}' if args.suffix_filter is not None else '*'),
                recursive=True
        )):
            # find target file
            target_file = os.path.join(args.gt_folder, os.path.basename(image_file))
            if not os.path.exists(target_file):
                filepath_without_suffix = os.path.splitext(target_file)[0]
                for path in glob.iglob(f'{filepath_without_suffix}.*'):
                    target_file = path
                    break
            if not os.path.exists(target_file):
                continue

            with rasterio.open(image_file) as dst:
                if self.input_size is None or (dst.height, dst.width) == self.input_size:
                    # not convert to tiles
                    self.item_ids.append((image_file, target_file, file_id, None))
                else:
                    # calculate how many input tiles that each image crop into
                    transform = ToGridTensor((dst.height, dst.width), self.input_size)
                    self.item_ids.extend([
                        (image_file, target_file, file_id, tile_id)
                        for tile_id in range(transform.cols * transform.rows)
                    ])

    def load_item(self, file: str, transform: Callable, use_grid: bool = False, **kwargs) -> torch.Tensor:

        with rasterio.open(file) as dst:
            data = dst.read(**kwargs)
            data = transform(data)
            # for x: treat padding area as nodata area
            # for y: if there is not dst.nodata, treat padding area as background which value is 0
            padding_value = dst.nodata or 0
            if use_grid:
                to_tile_tensor = ToGridTensor(
                    image_size=data.shape[-2:], input_size=self.input_size, padding_value=padding_value
                )

                data = to_tile_tensor(data)

            return data

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # important: when the shuffle argument in dataloader is True, index is not continuous in called
        x_file, y_file, file_id, tile_id = self.item_ids[index]

        # image size equals input size
        if tile_id is None:
            x = self.load_item(x_file, self.image_transform, use_grid=False)
            y = self.load_item(y_file, self.gt_transform, use_grid=False)
            return x, y.squeeze(0)

        # this file have been cached
        if file_id == self.cached_file_id:
            return self.item_cache[0][tile_id], self.item_cache[1][tile_id].squeeze(0)

        # cache x,y tiles for new file_id
        self.cached_file_id = file_id
        x_tiles = self.load_item(x_file, self.image_transform, use_grid=True, masked=True)
        y_tiles = self.load_item(y_file, self.gt_transform, use_grid=True)
        self.item_cache = [x_tiles, y_tiles]
        return self.item_cache[0][tile_id], self.item_cache[1][tile_id].squeeze(0)

    def __len__(self) -> int:
        """ Return the length of the dataset """
        return len(self.item_ids)
