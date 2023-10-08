from torch.utils.data import Dataset
import dataclasses
from typing import Literal
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import torch
import json


@dataclasses.dataclass
class DatasetArgs:
    """
    folder: Path to the dataset
    folds: List of ints specifying which of the 5 official folds to load.
        By default, (when None is specified) all folds are loaded.
    task: 'semantic' or 'instance'. Defines which type of target is returned by the dataloader.
    norm: If true, images are standardised using pre-computed channel-wise means and standard deviations.
    cache: If True, the loaded samples stay in RAM, default False.
    mem16: Additional arg for cache. If True, the image time series tensors
        stored in half precision in RAM for efficiency.
        They are cast back to float32 when returned by __getitem__.
    sats: defines the satellites to use (only Sentinel-2 is available in v1.0)
    """
    folder: str
    folds: list[int] = None
    task: Literal["semantic", "instance"] = "semantic"
    norm: bool = True
    cache: bool = False
    mem16: bool = False
    sats: tuple[Literal["S1", "S2"]] = ("S2",)


def date2days(x: int) -> int:
    """
    Args:
        x: int value like 20180202

    Returns:
        days from the first day in year
    """
    date = datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
    first_day = datetime.date(date.year, 1, 1)
    days = (date - first_day).days
    # assume every year is 365 day, days is in 0-364
    return days if days <= 364 else 364


class PastisDataset(Dataset):
    """ Pytorch Dataset class to load samples from the PASTIS dataset, for semantic and instance segmentation.
    The Dataset yields ((data, dates), gt) tuples, where:
        - data contains the image time series
        - dates contains the date sequence of the observations expressed in Gregorian calendar, 0 for padding data
        - gt is the semantic or instance ground truth
            - If 'semantic' the gt tensor is a tensor containing the class of each pixel.
            - If 'instance' the target tensor is the concatenation of several signals,
            necessary to train the Parcel-as-Points module:
                - the centerness heatmap,
                - the instance ids,
                - the voronoi partitioning of the patch regards to the parcels' centers,
                - the (height, width) size of each parcel
                - the semantic label of each parcel
                - the semantic label of each pixel
    """

    def __init__(self, args: DatasetArgs):
        super().__init__()
        self.folder = args.folder
        self.folds = args.folds or range(1, 6)
        self.norm = args.norm
        self.task = args.task
        self.cache = args.cache
        self.mem16 = args.mem16
        self.sats = args.sats

        self.memory = {}

        # Get metadata
        print("Reading patch metadata . . .")
        self.meta_table = gpd.read_file(os.path.join(args.folder, "metadata.geojson"))
        self.meta_table.index = self.meta_table["ID_PATCH"].astype(int)
        self.meta_table.sort_index(inplace=True)

        self.date_tables = {s: None for s in args.sats}

        # all possible dates
        for s in args.sats:
            dates = self.meta_table[f"dates-{s}"]
            sat_dates_dict = {}
            for patch_id, date_seq in dates.items():
                d = pd.DataFrame().from_dict(date_seq, orient="index")
                # d: DataFrame(0:20180101, 1:20180601, 2:...)
                # d[0]: Series(0:20180101, 1:20180601, 2:...)
                d = d[0].apply(date2days)
                # date table is consist of 0 and 1
                sat_dates_dict[patch_id] = d.values
            self.date_tables[s] = sat_dates_dict

        print("Done.")

        # Select Fold samples
        self.meta_table = pd.concat(
            [self.meta_table[self.meta_table["Fold"] == f] for f in self.folds]
        )

        self.len = self.meta_table.shape[0]
        self.patch_ids = self.meta_table.index

        # Get normalisation values
        if args.norm:
            self.norm = {}
            for s in self.sats:
                with open(os.path.join(self.folder, f"NORM_{s}_patch.json"), "r") as file:
                    norm_vals = json.loads(file.read())
                means = [norm_vals[f"Fold_{f}"]["mean"] for f in self.folds]
                stds = [norm_vals[f"Fold_{f}"]["std"] for f in self.folds]
                self.norm[s] = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
                self.norm[s] = (
                    torch.from_numpy(self.norm[s][0]).float(),
                    torch.from_numpy(self.norm[s][1]).float(),
                )
        else:
            self.norm = None
        print("Dataset ready.")

    def __len__(self):
        return self.len

    def get_dates(self, patch_id: int):
        """
        Returns:
            x: {s: torch.Tensor (t) } in float32
        """
        dates = {s: torch.from_numpy(self.date_tables[s][patch_id]) for s in self.sats}
        return dates

    def get_x(self, patch_id: int) -> dict:
        """
        Returns:
            x: {s: torch.Tensor (t,c,h,w) } in float32
        """
        data = {
            satellite: np.load(
                os.path.join(
                    self.folder,
                    f"DATA_{satellite}",
                    f"{satellite}_{patch_id}.npy",
                )
            ).astype(np.float32)
            for satellite in self.sats
        }  # T x C x H x W arrays
        data = {s: torch.from_numpy(a) for s, a in data.items()}

        if self.norm is not None:
            data = {
                s: (d - self.norm[s][0][None, :, None, None])  # move channel dimension in mean and std
                   / self.norm[s][1][None, :, None, None]
                for s, d in data.items()
            }
        return data

    def get_y(self, patch_id: int) -> torch.Tensor:
        """
        Returns:
            if task is semantic: torch.Tensor (h,w)
            if task is instance:
        """
        if self.task == "semantic":
            y = np.load(
                os.path.join(
                    self.folder, "ANNOTATIONS", f"TARGET_{patch_id}.npy"
                )
            )
            y = torch.from_numpy(y[0].astype(int))
        elif self.task == "instance":
            heatmap = np.load(
                os.path.join(
                    self.folder,
                    "INSTANCE_ANNOTATIONS",
                    "HEATMAP_{}.npy".format(patch_id),
                )
            )

            instance_ids = np.load(
                os.path.join(
                    self.folder,
                    "INSTANCE_ANNOTATIONS",
                    "INSTANCES_{}.npy".format(patch_id),
                )
            )
            pixel_to_object_mapping = np.load(
                os.path.join(
                    self.folder,
                    "INSTANCE_ANNOTATIONS",
                    "ZONES_{}.npy".format(patch_id),
                )
            )

            pixel_semantic_annotation = np.load(
                os.path.join(
                    self.folder, "ANNOTATIONS", "TARGET_{}.npy".format(patch_id)
                )
            )

            size = np.zeros((*instance_ids.shape, 2))
            object_semantic_annotation = np.zeros(instance_ids.shape)
            for instance_id in np.unique(instance_ids):
                if instance_id != 0:
                    h = (instance_ids == instance_id).any(axis=-1).sum()
                    w = (instance_ids == instance_id).any(axis=-2).sum()
                    size[pixel_to_object_mapping == instance_id] = (h, w)
                    object_semantic_annotation[
                        pixel_to_object_mapping == instance_id
                        ] = pixel_semantic_annotation[instance_ids == instance_id][0]

            y = torch.from_numpy(
                np.concatenate(
                    [
                        heatmap[:, :, None],  # 0
                        instance_ids[:, :, None],  # 1
                        pixel_to_object_mapping[:, :, None],  # 2
                        size,  # 3-4
                        object_semantic_annotation[:, :, None],  # 5
                        pixel_semantic_annotation[:, :, None],  # 6
                    ],
                    axis=-1,
                )
            ).float()
        else:
            raise Exception
        return y

    def __getitem__(self, index):
        """
        important: you can modify the return value, though they be cached. because pad_collate function return
            new object ( add b dimension )
        Returns:
            (x,dates), y
        """
        patch_id = self.patch_ids[index]

        # have cached
        if self.cache and patch_id in self.memory.keys():
            (x, dates), y = self.memory[patch_id]

            if self.mem16:
                x = {k: v.float() for k, v in x.items()}

        else:
            x = self.get_x(patch_id)
            dates = self.get_dates(patch_id)
            y = self.get_y(patch_id)

            if self.cache:
                if self.mem16:
                    cached_x = {k: v.half() for k, v in x.items()}
                    self.memory[patch_id] = [(cached_x, dates), y]
                else:
                    self.memory[patch_id] = [(x, dates), y]

        # not return dict if there is only one sat
        if len(self.sats) == 1:
            x = x[self.sats[0]]
            dates = dates[self.sats[0]]

        return (x, dates), y, patch_id
