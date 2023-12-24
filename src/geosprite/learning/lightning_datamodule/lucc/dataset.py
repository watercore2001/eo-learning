import dataclasses
import os

import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset
import glob
from einops import rearrange
from torchvision.transforms import v2


from .mask_generator import MaskGenerator

__all__ = ["LuccBaseDatasetArgs",
           "LuccPretrainDatasetArgs", "LuccPretrainDataset",
           "LuccFineTuningDatasetArgs", "LuccFineTuningDataset",
           "LuccPredictDatasetArgs", "LuccPredictDataset"]


@dataclasses.dataclass(kw_only=True)
class LuccBaseDatasetArgs:
    image_size: int
    model_patch_size: int
    # do not need input
    bands: list[str] = None
    norm_min: np.ndarray = None
    norm_max: np.ndarray = None


class LuccBaseDataset(Dataset):
    available_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

    def __init__(self, args: LuccBaseDatasetArgs):
        self.bands = args.bands
        self.image_size = args.image_size
        self.model_patch_size = args.model_patch_size
        self.norm_min = args.norm_min
        self.norm_max = args.norm_max

        self.image_size_in_patch_unit = self.image_size // self.model_patch_size
        mask = np.zeros(shape=(len(self.available_bands), self.image_size_in_patch_unit, self.image_size_in_patch_unit))
        missing_indices = [i for i, band in enumerate(self.available_bands) if band not in self.bands]
        mask[missing_indices, :, :] = 1
        self.missing_band_mask = torch.Tensor(mask)

    def image_transform(self, x: np.ndarray) -> torch.Tensor:
        all_shape = (len(self.available_bands),) + x.shape[1:]
        all_data = np.zeros(shape=all_shape)

        band_indices = [self.available_bands.index(band) for band in self.bands]
        all_data[band_indices, :, :] = x

        all_data = np.clip((all_data - self.norm_min[:, None, None])
                           / ((self.norm_max - self.norm_min)[:, None, None]),
                           a_min=0, a_max=1)

        return torch.Tensor(all_data)


class LuccFileDataset(LuccBaseDataset):
    def __init__(self, args: LuccBaseDatasetArgs):
        super().__init__(args)
        self.rel_paths = []

    def __len__(self):
        return len(self.rel_paths)

    def get_x(self, item_path: str):
        with rasterio.open(item_path) as src:
            data = src.read()

        return self.image_transform(data)

    @staticmethod
    def get_y(item_path: str):
        with rasterio.open(item_path) as src:
            data = src.read()
        data = rearrange(data, pattern="1 h w -> h w")
        return torch.tensor(data, dtype=torch.long)


@dataclasses.dataclass(kw_only=True)
class LuccPretrainDatasetArgs(LuccBaseDatasetArgs):
    """
    folders: pretrain image is distributed in multi folder
    mask_patch_size: mask patch size
    mask_ratio: mask ratio
    """
    folders: list[str]
    mask_patch_size: int
    mask_ratio: float
    use_aug: bool


class LuccPretrainDataset(LuccFileDataset):
    def __init__(self, args: LuccPretrainDatasetArgs):
        super().__init__(args)
        self.folders = args.folders
        self.use_aug = args.use_aug
        self.folder_ids, self.rel_paths = self.init_item_paths()
        self.mask_generator = MaskGenerator(image_size=args.image_size,
                                            channels=len(self.available_bands),
                                            mask_patch_size=args.mask_patch_size,
                                            model_patch_size=args.model_patch_size,
                                            mask_ratio=args.mask_ratio)

    def image_transform(self, x: np.ndarray) -> torch.Tensor:
        all_data = super().image_transform(x)
        if self.use_aug:
            aug_transform = v2.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5)
            ])
            all_data = aug_transform(all_data)
        return all_data

    def init_item_paths(self) -> tuple[list, list]:
        # print("start init item paths.")
        folder_ids = []
        rel_paths = []
        for i, folder in enumerate(self.folders):
            sub_paths = glob.glob("*/*.tif", root_dir=folder)
            folder_ids.extend([i] * len(sub_paths))
            rel_paths.extend(sub_paths)
        # print("finish init item paths.")
        return folder_ids, rel_paths

    def __getitem__(self, index: int):
        i = self.folder_ids[index]
        rel_path = self.rel_paths[index]
        item_path = os.path.join(self.folders[i], rel_path)

        x = self.get_x(item_path)
        mask = self.mask_generator()
        # set missing band as mask
        missing_indices = [i for i, band in enumerate(self.available_bands) if band not in self.bands]
        mask[missing_indices, :, :] = 1

        # x shape: c h w
        # mask shape: c patch_num_in_h patch_num_in_w
        return {"x": x, "mask": mask, "tif_path": item_path}


@dataclasses.dataclass(kw_only=True)
class LuccFineTuningDatasetArgs(LuccBaseDatasetArgs):
    folder: str


class LuccFineTuningDataset(LuccFileDataset):
    sat_name = "sat"
    gt_name = "gt"

    def __init__(self, args: LuccFineTuningDatasetArgs):
        super().__init__(args)
        self.sat_folder = os.path.join(args.folder, self.sat_name)
        self.gt_folder = os.path.join(args.folder, self.gt_name)
        self.image_size_in_patch_unit = self.image_size // self.model_patch_size
        self.item_paths = self.init_item_paths()

    def init_item_paths(self) -> list:
        item_paths = []

        for rel_path in glob.glob(pathname=f"*/*.tif", root_dir=self.sat_folder):
            sat_path = os.path.join(self.sat_folder, rel_path)
            gt_path = os.path.join(self.gt_folder, rel_path)
            item_paths.append((sat_path, gt_path))
        return item_paths

    def __getitem__(self, index: int):
        sat_path, gt_path = self.item_paths[index]

        x = self.get_x(sat_path)
        y = self.get_y(gt_path)

        # x and y shape: c h w
        return {"x": x, "mask": self.missing_band_mask, "y": y}


@dataclasses.dataclass(kw_only=True)
class LuccPredictDatasetArgs(LuccBaseDatasetArgs):
    """
    image_arrays: Path to the dataset
    """
    image_arrays: list[np.ndarray]


class LuccPredictDataset(LuccBaseDataset):

    def __init__(self, args: LuccPredictDatasetArgs):
        super().__init__(args)

        self.image_arrays = args.image_arrays

    def get_x(self, index: int):
        data = self.image_arrays[index]
        return self.image_transform(data)

    def __len__(self):
        return len(self.image_arrays)

    def __getitem__(self, index: int):
        x = self.get_x(index)

        return {"x": x, "mask": self.missing_band_mask}















