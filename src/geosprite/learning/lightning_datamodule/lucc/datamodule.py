from pytorch_lightning import LightningDataModule
import dataclasses
import os

from torch.utils.data import DataLoader
from .dataset import *
from ..base_dataloader import BaseDataloaderArgs

__all__ = ["LuccPretrainDataModule", "LuccFineTuningDataModule"]


class LuccPretrainDataModule(LightningDataModule):
    def __init__(self,
                 dataset_args: PretrainDatasetArgs,
                 dataloader_args: BaseDataloaderArgs):
        super().__init__()
        self.dataset_args = dataset_args
        self.dataloader_args = dataloader_args

        self.train_dataset = None
        self.predict_dataset = None

    def make_dataset(self, sub_folder_name: str):
        sub_folder = os.path.join(self.dataset_args.folder, sub_folder_name)
        dataset_args = dataclasses.replace(self.dataset_args, folder=sub_folder)
        return LuccPretrainDataset(dataset_args)

    def setup(self, stage: [str] = None):
        if stage == "fit":
            self.train_dataset = self.make_dataset("train")
        if stage == "predict":
            self.predict_dataset = self.make_dataset("predict")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **dataclasses.asdict(self.dataloader_args))

    def predict_dataloader(self):
        dataloader_args = dataclasses.replace(self.dataloader_args, shuffle=False)
        return DataLoader(self.predict_dataset, **dataclasses.asdict(dataloader_args))


class LuccFineTuningDataModule(LightningDataModule):
    def __init__(self,
                 dataset_args: FineTuningDatasetArgs,
                 dataloader_args: BaseDataloaderArgs):
        super().__init__()
        self.dataset_args = dataset_args
        self.dataloader_args = dataloader_args

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def make_dataset(self, sub_folder_name: str):
        sub_folder = os.path.join(self.dataset_args.folder, sub_folder_name)
        dataset_args = dataclasses.replace(self.dataset_args, folder=sub_folder)
        return LuccFineTuningDataset(dataset_args)

    def setup(self, stage: [str] = None):
        if stage == "fit":
            self.train_dataset = self.make_dataset("train")
            self.val_dataset = self.make_dataset("val")
        if stage == "test":
            self.test_dataset = self.make_dataset("test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **dataclasses.asdict(self.dataloader_args))

    def val_dataloader(self):
        dataloader_args = dataclasses.replace(self.dataloader_args, shuffle=False)
        return DataLoader(self.val_dataset, **dataclasses.asdict(dataloader_args))

    def test_dataloader(self):
        dataloader_args = dataclasses.replace(self.dataloader_args, shuffle=False)
        return DataLoader(self.test_dataset, **dataclasses.asdict(dataloader_args))


