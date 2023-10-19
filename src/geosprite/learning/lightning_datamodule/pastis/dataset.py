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
from einops import repeat


@dataclasses.dataclass
class DatasetArgs:
    """
    folder: Path to the dataset
    folds: List of ints specifying which of the 5 official folds to load.
        By default, (when None is specified) all folds are loaded.
    task: 'semantic' or 'instance'. Defines which type of target is returned by the dataloader.
    use_norm: If true, images are standardised using pre-computed channel-wise means and standard deviations.
    use_location: If True, use location
    """
    folder: str
    folds: list[int] = None
    task: Literal["semantic", "instance"] = "semantic"
    use_norm: bool = True
    use_location: bool = False


def date2days(x: int) -> int:
    """
    Args:
        x: int value like 20180202

    Returns:
        day of year, 0 means the first day
    """
    date = datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
    first_day = datetime.date(date.year, 1, 1)
    day_of_year = (date - first_day).days
    return day_of_year


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
        self.task = args.task
        self.use_norm = args.use_norm
        self.use_location = args.use_location

        # Get metadata
        print("Reading patch metadata . . .")
        self.meta_table = gpd.read_file(os.path.join(args.folder, "metadata.geojson"))
        self.meta_table.index = self.meta_table["ID_PATCH"].astype(int)
        self.meta_table.sort_index(inplace=True)

        # all possible dates
        dates = self.meta_table[f"dates-S2"]
        sat_dates_dict = {}
        for patch_id, date_seq in dates.items():
            d = pd.DataFrame().from_dict(date_seq, orient="index")
            # d: DataFrame(0:20180101, 1:20180601, 2:...)
            # d[0]: Series(0:20180101, 1:20180601, 2:...)
            d = d[0].apply(date2days)
            # date table is consist of 0 and 1
            sat_dates_dict[patch_id] = d.values
        self.date_tables = sat_dates_dict

        print("Done.")

        # Select Fold samples
        self.meta_table = pd.concat(
            [self.meta_table[self.meta_table["Fold"] == f] for f in self.folds]
        )

        self.len = self.meta_table.shape[0]
        self.patch_ids = self.meta_table.index

        # Get normalisation values
        if args.use_norm:
            with open(os.path.join(self.folder, f"NORM_S2_patch.json"), "r") as file:
                norm_vals = json.loads(file.read())
            means = [norm_vals[f"Fold_{f}"]["mean"] for f in self.folds]
            stds = [norm_vals[f"Fold_{f}"]["std"] for f in self.folds]
            self.band_norm = (
                # mean across folds
                torch.from_numpy(np.stack(means).mean(axis=0)).float(),
                torch.from_numpy(np.stack(stds).mean(axis=0)).float(),)

        # Get Location norms
        if args.use_location:
            self.x_mean, self.x_std, self.y_mean, self.y_std = self.get_location_norm()

        print("Dataset ready.")

    def __len__(self):
        return self.len

    def get_location_norm(self):
        # x_sum, x_square_sum = 0, 0
        # y_sum, y_square_sum = 0, 0
        # for patch_id in self.patch_ids:
        #     bounds = self.meta_table.bounds.loc[patch_id]
        #     x = np.linspace(bounds.minx, bounds.maxx, 128)
        #     y = np.linspace(bounds.miny, bounds.maxy, 128)
        #     x_sum += x.mean(axis=0)
        #     x_square_sum += (x ** 2).mean(axis=0)
        #     y_sum += y.mean(axis=0)
        #     y_square_sum += (y ** 2).mean(axis=0)
        # x_mean = x_sum / self.len
        # x_std = (x_square_sum / self.len - x_mean ** 2) ** 0.5
        # y_mean = y_sum / self.len
        # y_std = (y_square_sum / self.len - y_mean ** 2) ** 0.5
        x_mean = 802104.9246430461
        x_std = 206516.56069328674
        y_mean = 6634731.626191179
        y_std = 219831.13482513756
        return x_mean, x_std, y_mean, y_std

    def get_dates(self, patch_id: int):
        """
        Returns:
            x: {s: torch.Tensor (t) } in float32
        """
        dates = torch.from_numpy(self.date_tables[patch_id])
        return dates

    def get_x(self, patch_id: int) -> dict:
        """
        Returns:
            x: {s: torch.Tensor (t,c,h,w) } in float32
        """
        # T x C x H x W
        data = torch.from_numpy(np.load(
            os.path.join(
                self.folder,
                f"DATA_S2",
                f"S2_{patch_id}.npy",
            )).astype(np.float32))

        if self.use_norm:
            data = (data - self.band_norm[0][None, :, None, None]) / self.band_norm[1][None, :, None, None]

        if self.use_location:
            t, c, h, w = data.size()
            bounds = self.meta_table.bounds.loc[patch_id]
            x = torch.from_numpy(np.linspace(bounds.minx, bounds.maxx, w).astype(np.float32))
            x = (x - self.x_mean) / self.x_std
            x = repeat(x, "w -> t 1 h w", t=t, h=h)

            y = torch.from_numpy(np.linspace(bounds.miny, bounds.maxy, h).astype(np.float32))
            y = (y - self.y_mean) / self.y_std
            y = repeat(y, "h -> t 1 h w", t=t, w=w)

            data = torch.cat([data, x, y], dim=1)

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
            y = torch.from_numpy(y[0].astype("int64"))
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

        x = self.get_x(patch_id)
        dates = self.get_dates(patch_id)
        y = self.get_y(patch_id)

        # patch_id for prediction
        return (x, dates), y
