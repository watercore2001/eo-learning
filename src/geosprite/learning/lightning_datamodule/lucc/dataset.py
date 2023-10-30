import dataclasses
import json
import os

import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset
import glob


@dataclasses.dataclass
class DatasetArgs:
    """
    folder: Path to the dataset
    use_norm: If true, images are standardised using pre-computed channel-wise min and max value.
    """
    folder: str
    use_norm: bool = True


class LuccDataset(Dataset):
    input_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    norm_filename = "norm.json"
    metadata_filename = "metadata.json"

    def __init__(self, args: DatasetArgs):
        self.folder = args.folder
        self.use_norm = args.use_norm

        self.item_paths = self.init_item_paths()
        self.norm_min, self.norm_max = self.init_min_max()

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
            min_list = [0 for _ in norm_data_dict.values()]
            max_list = [value["max"] for value in norm_data_dict.values()]
        return np.array(min_list), np.array(max_list)

    def get_missing_band_indices(self, item_path: str):
        metadata_path = os.path.join(os.path.dirname(item_path), self.metadata_filename)
        with open(metadata_path) as file:
            metadata = json.load(file)
        bands = metadata["bands"]
        missing_indices = [i for i, band in enumerate(self.input_bands) if band not in bands]
        return missing_indices

    def __len__(self):
        return len(self.item_paths)

    def __getitem__(self, index: int):
        item_path = self.item_paths[index]
        with rasterio.open(item_path) as src:
            data = src.read()

        missing_indices = self.get_missing_band_indices(item_path)
        data = np.insert(data, missing_indices, 0, axis=0)

        if self.use_norm:
            data = np.clip((data-self.norm_min[:, None, None])/((self.norm_max-self.norm_min)[:, None, None]), 0, 1)

        return torch.from_numpy(data), missing_indices





