# Copyright (c) GeoSprite. All rights reserved.
#
# Author: Jia Song
#

import collections.abc
import os
import re
from typing import List, Dict, AnyStr, Optional

import numpy as np
import rasterio
import torch
from osgeo import gdal


# def show_feature_map(feature_map):
#     import numpy as np
#     import scipy
#     import matplotlib.pyplot as plt
#
#     feature_map = feature_map.squeeze(0)
#     feature_map = feature_map.cpu().numpy()
#     feature_map_num = feature_map.shape[0]
#     row_num = np.ceil(np.sqrt(feature_map_num))
#     plt.figure()
#     for index in range(1, feature_map_num + 1):
#         plt.subplot(row_num, row_num, index)
#         plt.imshow(feature_map[index - 1], cmap='gray')
#         plt.axis('off')
#         scipy.misc.imsave(str(index) + ".png", feature_map[index - 1])
#     plt.show()


def save_as_image(path: str, data: np.ndarray, transparent: Optional[bool] = False) -> None:
    from PIL import Image
    import numpy

    arr = data * 255

    arr_rgb = numpy.stack(
        (arr.astype(numpy.uint8), numpy.zeros(arr.shape, dtype=numpy.uint8), numpy.zeros(arr.shape, dtype=numpy.uint8)))

    arr_rgb = arr_rgb.transpose((1, 2, 0))

    if transparent:
        img = Image.fromarray(arr_rgb).convert('RGBA')

        width, height = img.size

        label_color = (255, 0, 0, 255)

        for h in range(height):
            for l in range(width):
                dot = (l, h)
                color = img.getpixel(dot)
                if color != label_color:
                    color = color[:-1] + (0,)
                    img.putpixel(dot, color)
    else:
        im = Image.fromarray(arr_rgb).convert('RGB')

    im.save(path)

    return path


def save_as_gtiff(pathname: str, data: np.ndarray, reference_image_file: str, color_entries: Optional[Dict] = None,
                  driver: Optional[str] = 'GTiff') -> str:
    if not isinstance(data, np.ndarray):
        raise RuntimeError(f"The type of the parameter 'data' should be a numpy-array type, but got '{type(data)}'")

    with rasterio.open(reference_image_file) as refer:
        profile = refer.profile
        nodata = profile.data['nodata']

        if nodata is not None:
            nodata_mask = refer.dataset_mask()
            data[nodata_mask == 0] = nodata

        dim = len(data.shape)
        window = ((0, data.shape[-2]), (0, data.shape[-1]))

        profile.update(
            driver='GTiff',
            compress='DEFLATE',
            dtype=data.dtype,
            width=data.shape[-1],
            height=data.shape[-2],
            photometric="RGB",
            count=data.shape[0] if dim > 2 else 1,
            win_transform=refer.window_transform(window)
        )

        def write_band(band_no, band_data) -> bool:
            _band = dst.GetRasterBand(band_no)

            if nodata is not None:
                _band.SetNoDataValue(nodata)
            if color_table is not None:
                _band.SetColorTable(color_table)

            _band.WriteArray(band_data)
            _band.FlushCache()
            _band.ComputeStatistics(False)

            return _band.ReadAsArray() is not None

        if color_entries is not None:
            color_table = gdal.ColorTable()

            for color in color_entries:
                if len(color) == 2:
                    if isinstance(color[1], list):
                        color = (color[0], tuple(color[1]))

                    color_table.SetColorEntry(*color)
        else:
            color_table = None

        if data.dtype == np.uint8:
            dtype = gdal.GDT_Byte
        elif data.dtype == np.uint16:
            dtype = gdal.GDT_UInt16
        elif data.dtype == np.int16:
            dtype = gdal.GDT_Int16
        elif data.dtype == np.uint32:
            dtype = gdal.GDT_UInt32
        elif data.dtype == np.int32:
            dtype = gdal.GDT_Int32
        elif data.dtype == np.float32:
            dtype = gdal.GDT_Float32
        elif data.dtype == np.float64:
            dtype = gdal.GDT_Float64
        else:
            dtype = gdal.GDT_Unknown

        dirname = os.path.dirname(pathname)
        os.makedirs(dirname, exist_ok=True)

        create_options = ["COMPRESS=LZW", "PREDICTOR=2", "TILED=YES", "NUM_THREADS=ALL_CPUS"]
        dim = len(data.shape)

        _driver = gdal.GetDriverByName(driver)
        dst = _driver.Create(pathname, data.shape[-1], data.shape[-2], data.shape[-3] if dim > 2 else 1, dtype,
                             options=create_options)

        if dst is None:
            raise RuntimeError(f"Create {driver} file '{pathname}' failed.")

        dst.SetProjection(profile.data['crs'].wkt)
        aff = tuple(profile.data['transform'])
        dst.SetGeoTransform(aff[2:3] + aff[0:2] + aff[5:6] + aff[3:5])
        nodata = profile.data['nodata']

        if nodata is not None:
            if gdal.GDT_Byte <= dtype <= gdal.GDT_Int32:
                nodata = int(nodata)
            elif gdal.GDT_Float32 <= dtype <= gdal.GDT_Float64:
                nodata = float(nodata)

        try:
            i = 1
            if dim > 2:
                for arr in data:

                    if write_band(i, arr) is False:
                        os.remove(pathname)
                        return None

                    i += 1
            else:
                write_band(i, data)

            return pathname
        except Exception as e:
            if os.path.exists(pathname):
                os.remove(pathname)
                import shutil
                shutil.rmtree(dirname, ignore_errors=True)

            return None


def writelines(lines: List[AnyStr], filename: str, dirname: Optional[str] = None) -> None:
    file = os.path.join(dirname, filename) if dirname is not None else filename

    os.makedirs(os.path.dirname(file), exist_ok=True)

    with open(file, "w+", encoding='utf-8') as f:
        f.writelines(lines)
