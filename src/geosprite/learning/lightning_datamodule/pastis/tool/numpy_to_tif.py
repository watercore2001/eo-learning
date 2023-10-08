from geosprite.learning.lightning_datamodule.pastis.dataset_for_predict import PastisDataset, DatasetArgs
import argparse
import geopandas as gpd
import os
import numpy as np
import rasterio
from rasterio import CRS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--pastis_folder", type=str)
    parser.add_argument("-i", "--input_folder", type=str)
    parser.add_argument("-o", "--output_folder", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    meta_table = gpd.read_file(os.path.join(args.pastis_folder, "metadata.geojson"))
    meta_table.index = meta_table["ID_PATCH"].astype(int)
    epsg_code = meta_table.crs.to_epsg()
    for patch_id in meta_table.index:
        input_filepath = os.path.join(args.input_folder, f"S2_{patch_id}.npy")
        y = np.load(input_filepath)[20, 2::-1, :, :]
        b, h, w = y.shape
        bounds = meta_table.bounds.loc[patch_id]
        transform = rasterio.transform.from_bounds(west=bounds.minx, south=bounds.miny,
                                                   east=bounds.maxx, north=bounds.maxy,
                                                   width=w, height=h)
        output_filepath = os.path.join(args.output_folder, f"x_{patch_id}.tif")
        with rasterio.open(fp=output_filepath, mode="w", width=w, height=h, count=3,
                           crs=CRS.from_epsg(epsg_code), transform=transform, dtype="int16") as dst:
            dst.write(y)


if __name__ == "__main__":
    main()
