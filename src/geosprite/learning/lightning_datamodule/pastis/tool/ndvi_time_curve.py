from geosprite.learning.lightning_datamodule.pastis.dataset_for_predict import PastisDataset, DatasetArgs
from torch.utils.data import DataLoader
import argparse
from einops import reduce, repeat, rearrange
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--pastis_folder", type=str)
    parser.add_argument("-o", "--output_folder", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = PastisDataset(DatasetArgs(folder=args.pastis_folder, folds=[1, 2, 3, 4, 5], norm=False))
    dataloader = DataLoader(dataset)
    for (x, dates), y, patch_id in dataloader:
        # x: b t c h w
        # d: b t
        # y: b h w
        red = x[:, :, 2, :, :]
        nir = x[:, :, 6, :, :]
        # b t h w
        ndvi = (nir - red) / (nir + red)
        ndvi = rearrange(ndvi, "b t h w -> t b h w")
        b, t, h, w = ndvi.size()

        # each crop type
        for i in range(20):
            crop_i_ndvi = ndvi[:, y == i]
            nan_mask = torch.isnan(crop_i_ndvi)
            crop_i_ndvi[nan_mask] = 0
            pixel_num_i_in_times = crop_i_ndvi.size(dim=1) - torch.sum(nan_mask, dim=1)
            ndvi_sum_i_in_times = torch.sum(crop_i_ndvi, dim=1)
            pass


if __name__ == "__main__":
    main()
