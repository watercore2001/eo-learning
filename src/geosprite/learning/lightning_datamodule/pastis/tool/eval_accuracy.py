import argparse
import os
import numpy as np
from geosprite.learning.lightning_datamodule.pastis.dataset_for_predict import PastisDataset, DatasetArgs
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, JaccardIndex
import pprint
from einops import rearrange
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--pastis_folder", type=str)
    parser.add_argument("-o", "--output_folder", type=str)
    return parser.parse_args()


def find_good_and_bad(my_dict):
    # 查找字典中最大的 5 个值以及它们对应的键
    largest_items = sorted(my_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    print("最大的五个值对应的：", largest_items)

    # 查找字典中最小的 5 个值以及它们对应的键
    smallest_items = sorted(my_dict.items(), key=lambda x: x[1])[:10]
    print("最小的五个值对应的：", smallest_items)


def main():
    args = parse_args()
    dataset = PastisDataset(DatasetArgs(folder=args.pastis_folder, folds=[5]))
    dataloader = DataLoader(dataset)
    metric_1 = JaccardIndex(task="multiclass", average="macro", num_classes=20, ignore_index=19)
    metric_2 = JaccardIndex(task="multiclass", average="macro", num_classes=20, ignore_index=19)
    metric_3 = JaccardIndex(task="multiclass", average="macro", num_classes=20, ignore_index=19)
    metric_4 = JaccardIndex(task="multiclass", average="macro", num_classes=20, ignore_index=19)
    metric_diction = {}
    for x, y, patch_id in dataloader:
        patch_id = patch_id.numpy()[0]
        pred_file = os.path.join(args.output_folder, f"{patch_id}.npy")
        pred = torch.Tensor(np.load(pred_file))
        pred = rearrange(pred, "h w->1 h w")
        if patch_id < 20000:
            metric_1.update(pred, y)
        elif patch_id < 30000:
            metric_2.update(pred, y)
        elif patch_id < 40000:
            metric_3.update(pred, y)
        elif patch_id < 50000:
            metric_4.update(pred, y)
        else:
            return
    print(metric_1.compute())
    print(metric_2.compute())
    print(metric_3.compute())
    print(metric_4.compute())
    # find_good_and_bad(metric_diction)


if __name__ == "__main__":
    main()
