import dataclasses
import json
import os

import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset
import glob
from einops import rearrange

from .mask_generator import MaskGenerator

__all__ = ["PretrainDatasetArgs", "LuccPretrainDataset", "FineTuningDatasetArgs", "LuccFineTuningDataset"]


class LuccBaseDataset(Dataset):
    def __init__(self):
        self.use_norm = True
        self.norm_min = None
        self.norm_max = None
        self.item_paths = []

    available_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    norm_filename = "norm.json"
    metadata_filename = "metadata.json"

    @staticmethod
    def init_min_max(norm_data_path: str) -> tuple:
        with open(norm_data_path) as file:
            norm_data_dict = json.load(file)
            # use 0 as min value
            min_list = [0 for _ in norm_data_dict.values()]
            max_list = [value["max"] for value in norm_data_dict.values()]
        return np.array(min_list), np.array(max_list)

    def get_bands(self, item_path: str):
        metadata_path = os.path.join(os.path.dirname(item_path), self.metadata_filename)
        with open(metadata_path) as file:
            metadata = json.load(file)
        bands = metadata["bands"]
        return bands

    def __len__(self):
        return len(self.item_paths)

    def get_x(self, item_path: str, bands: list[str]):
        with rasterio.open(item_path) as src:
            data = src.read()

        all_shape = (len(self.available_bands),) + data.shape[1:]
        all_data = np.zeros(shape=all_shape)

        band_indices = [self.available_bands.index(band) for band in bands]
        all_data[band_indices, :, :] = data

        if self.use_norm:
            all_data = np.clip((all_data - self.norm_min[:, None, None])
                               / ((self.norm_max - self.norm_min)[:, None, None]),
                               a_min=0, a_max=1)

        return torch.Tensor(all_data)

    @staticmethod
    def get_y(item_path):
        with rasterio.open(item_path) as src:
            data = src.read()
        data = rearrange(data, pattern="1 h w -> h w")
        return torch.tensor(data, dtype=torch.long)


@dataclasses.dataclass
class PretrainDatasetArgs:
    """
    folder: Path to the dataset
    image_size: model input image size
    channels: model input channels
    mask_patch_size: mask patch size
    model_patch_size: model patch size
    mask_ratio: mask ratio
    use_norm: If true, images are standardised using pre-computed channel-wise min and max value.
    """
    folder: str
    image_size: int
    mask_patch_size: int
    model_patch_size: int
    mask_ratio: float
    use_norm: bool = True


class LuccPretrainDataset(LuccBaseDataset):
    def __init__(self, args: PretrainDatasetArgs):
        super().__init__()
        self.folder = args.folder
        self.use_norm = args.use_norm

        self.item_paths = self.init_item_paths()
        norm_data_path = os.path.join(self.folder, self.norm_filename)
        self.norm_min, self.norm_max = self.init_min_max(norm_data_path)

        self.mask_generator = MaskGenerator(image_size=args.image_size,
                                            channels=len(self.available_bands),
                                            mask_patch_size=args.mask_patch_size,
                                            model_patch_size=args.model_patch_size,
                                            mask_ratio=args.mask_ratio)

    def init_item_paths(self) -> list:
        item_paths = []
        for scene_rel_folder in os.listdir(self.folder):
            scene_folder = os.path.join(self.folder, scene_rel_folder)
            if not os.path.isdir(scene_folder):
                continue
            item_paths.extend(glob.glob(os.path.join(scene_folder, "*.tif")))
        return item_paths

    def __getitem__(self, index: int):
        item_path = self.item_paths[index]
        bands = self.get_bands(item_path)

        x = self.get_x(item_path, bands)

        mask = self.mask_generator()
        # set missing band as mask
        missing_indices = [i for i, band in enumerate(self.available_bands) if band not in bands]
        mask[missing_indices, :, :] = 1

        # x shape: c h w
        # mask shape: c patch_num_in_h patch_num_in_w
        return {"x": x, "mask": mask, "tif_path": item_path}


@dataclasses.dataclass
class FineTuningDatasetArgs:
    """
    folder: Path to the dataset
    image_size: model input image size
    model_patch_size: model patch size
    use_norm: If true, images are standardised using pre-computed channel-wise min and max value.
    """
    folder: str
    image_size: int
    model_patch_size: int
    use_norm: bool = True


class LuccFineTuningDataset(LuccBaseDataset):
    sat_name = "sat"
    gt_name = "gt"

    def __init__(self, args: FineTuningDatasetArgs):
        super().__init__()
        self.folder = args.folder
        self.sat_folder = os.path.join(args.folder, self.sat_name)
        self.gt_folder = os.path.join(args.folder, self.gt_name)

        self.image_size = args.image_size
        self.model_patch_size = args.model_patch_size
        self.image_size_in_patch_unit = self.image_size // self.model_patch_size

        self.use_norm = args.use_norm

        self.item_paths = self.init_item_paths()
        norm_data_path = os.path.join(self.sat_folder, self.norm_filename)
        self.norm_min, self.norm_max = self.init_min_max(norm_data_path)

    def init_item_paths(self) -> list:
        item_paths = []

        for rel_path in glob.glob(pathname=f"*/*.tif", root_dir=self.sat_folder):
            sat_path = os.path.join(self.sat_folder, rel_path)
            gt_path = os.path.join(self.gt_folder, rel_path)
            item_paths.append((sat_path, gt_path))
        return item_paths

    def __getitem__(self, index: int):
        sat_path, gt_path = self.item_paths[index]

        bands = self.get_bands(sat_path)
        x = self.get_x(sat_path, bands)

        mask = np.zeros(shape=(len(self.available_bands), self.image_size_in_patch_unit, self.image_size_in_patch_unit))
        missing_indices = [i for i, band in enumerate(self.available_bands) if band not in bands]
        mask[missing_indices, :, :] = 1
        mask = torch.Tensor(mask)

        y = self.get_y(gt_path)

        # x and y shape: c h w
        return {"x": x, "mask": mask, "y": y}











