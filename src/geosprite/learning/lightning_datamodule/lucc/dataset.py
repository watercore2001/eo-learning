import dataclasses
import json
import os

import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset
import glob

from .mask_generator import MaskGenerator


@dataclasses.dataclass
class DatasetArgs:
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


class LuccDataset(Dataset):
    available_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    norm_filename = "norm.json"
    metadata_filename = "metadata.json"

    def __init__(self, args: DatasetArgs):
        self.folder = args.folder
        self.use_norm = args.use_norm

        self.item_paths = self.init_item_paths()
        self.norm_min, self.norm_max = self.init_min_max()

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

    def init_min_max(self) -> tuple:
        norm_data_path = os.path.join(self.folder, self.norm_filename)
        with open(norm_data_path) as file:
            norm_data_dict = json.load(file)
            # use 0 as min value
            min_list = [0 for _ in norm_data_dict.values()]
            max_list = [value["max"] for value in norm_data_dict.values()]
        return np.array(min_list), np.array(max_list)

    def get_missing_band_indices(self, item_path: str):
        metadata_path = os.path.join(os.path.dirname(item_path), self.metadata_filename)
        with open(metadata_path) as file:
            metadata = json.load(file)
        bands = metadata["bands"]
        missing_indices = [i for i, band in enumerate(self.available_bands) if band not in bands]
        return missing_indices

    def get_x(self, item_path: str, missing_indices: list[int]):

        with rasterio.open(item_path) as src:
            data = src.read()

        data = np.insert(data, missing_indices, values=0, axis=0)
        if self.use_norm:
            data = np.clip((data - self.norm_min[:, None, None]) / ((self.norm_max - self.norm_min)[:, None, None]),
                           a_min=0, a_max=1)

        return torch.Tensor(data)

    def __len__(self):
        return len(self.item_paths)

    def __getitem__(self, index: int):
        item_path = self.item_paths[index]
        missing_indices = self.get_missing_band_indices(item_path)

        x = self.get_x(item_path, missing_indices)

        mask = self.mask_generator()
        # set missing band as mask
        mask[missing_indices, :, :] = 1

        # x shape: c h w
        # mask shape: c patch_num_in_h patch_num_in_w
        return {"x": x, "mask": mask, "tif_path": item_path}





