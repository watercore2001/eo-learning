import torch
import collections.abc
from torch.nn import functional
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from .dataset import PastisDataset, DatasetArgs
import dataclasses
from typing import Callable
from ..base_dataloader import BaseDataloaderArgs


@dataclasses.dataclass
class DataloaderArgs(BaseDataloaderArgs):
    collate_fn: Callable

    def __post_init__(self):
        self.collate_fn = lambda x: pad_collate(x, pad_value=0)


def pad_tensor(x, length, pad_value=0):
    pad_len = length - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, pad_len]
    return functional.pad(x, pad=pad, value=pad_value)


def pad_collate(batch, pad_value=0):
    # modified default_collate from the official pytorch repo
    # https://github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/collate.py
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        if len(elem.shape) > 0:
            # size: time series length for each item in batch
            sizes = [e.shape[0] for e in batch]
            m = max(sizes)
            if not all(s == m for s in sizes):
                # pad tensors which have a temporal dimension
                batch = [pad_tensor(e, m, pad_value=pad_value) for e in batch]
        return torch.stack(batch, dim=0)
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [pad_collate(samples, pad_value) for samples in transposed]

    raise TypeError("Format not managed : {}".format(elem_type))


class PastisDataModule(LightningDataModule):
    def __init__(self,
                 dataset_args: DatasetArgs,
                 dataloader_args: DataloaderArgs,
                 folds_split: list[list[int]] = None
                 ):
        super().__init__()
        self.dataset_args = dataset_args
        self.dataloader_args = dataloader_args
        self.train_folds, self.test_folds, self.val_folds = folds_split or [[1, 2, 3], [4], [5]]

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def make_dataset(self, folds: list[int]):
        update_args = {"folds": folds}
        dataset_args = dataclasses.replace(self.dataset_args, **update_args)
        # updates args
        dataset = PastisDataset(dataset_args)
        assert len(dataset) > 0, f"Dataset folder is empty. " \
                                 f"folder is {dataset_args.folder}. "
        return dataset

    def setup(self, stage: [str] = None) -> None:
        if stage == "fit":
            self.train_dataset = self.make_dataset(folds=self.train_folds)
            self.val_dataset = self.make_dataset(folds=self.val_folds)
        elif stage == "validate":
            self.val_dataset = self.make_dataset(folds=self.val_folds)
        elif stage == "test":
            self.test_dataset = self.make_dataset(folds=self.test_folds)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, **dataclasses.asdict(self.dataloader_args))

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, **dataclasses.asdict(self.dataloader_args))

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, **dataclasses.asdict(self.dataloader_args))
