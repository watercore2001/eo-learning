from pytorch_lightning import LightningDataModule
import dataclasses
import os
import json
import numpy as np
from typing import Literal

from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader
from .dataset import *
from ..base_dataloader import BaseDataloaderArgs

__all__ = ["LuccPretrainDataModule", "LuccFineTuningDataModule"]


class LuccBaseDataModule(LightningDataModule):
    metadata_filename = "metadata.json"

    def __init__(self, dataset_args: LuccBaseDatasetArgs, metadata_path: str):
        super().__init__()
        with open(metadata_path) as file:
            metadata = json.load(file)
            dataset_args.bands = metadata["bands"]
            norm_data_dict = metadata["norm"]
            dataset_args.norm_min = np.array([0 for _ in norm_data_dict.values()])
            dataset_args.norm_max = np.array([value["max"] for value in norm_data_dict.values()])


class LuccPretrainDataModule(LuccBaseDataModule):
    def __init__(self,
                 dataset_args: LuccPretrainDatasetArgs,
                 dataloader_args: BaseDataloaderArgs):
        metadata_path = os.path.join(dataset_args.folders[0], self.metadata_filename)
        super().__init__(dataset_args, metadata_path)

        self.dataset_args = dataset_args
        self.dataloader_args = dataloader_args

        self.train_dataset = None
        self.val_dataset = None
        self.predict_dataset = None

    def make_dataset(self, stage_name: Literal["train", "val", "test", "predict"]):
        stage_folders = []
        for folder in self.dataset_args.folders:
            stage_folder = os.path.join(folder, stage_name)
            if os.path.isdir(stage_folder):
                stage_folders.append(stage_folder)

        dataset_args = dataclasses.replace(self.dataset_args, folders=stage_folders)
        return LuccPretrainDataset(dataset_args)

    def setup(self, stage: [str] = None):
        if stage == "fit":
            self.train_dataset = self.make_dataset("train")
            self.val_dataset = self.make_dataset("val")
        if stage == "predict":
            self.predict_dataset = self.make_dataset("predict")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **dataclasses.asdict(self.dataloader_args))

    def val_dataloader(self):
        dataloader_args = dataclasses.replace(self.dataloader_args, shuffle=False)
        return DataLoader(self.val_dataset, **dataclasses.asdict(dataloader_args))

    def predict_dataloader(self):
        dataloader_args = dataclasses.replace(self.dataloader_args, shuffle=False)
        return DataLoader(self.predict_dataset, **dataclasses.asdict(dataloader_args))


class LuccFineTuningDataModule(LuccBaseDataModule):
    def __init__(self,
                 dataset_args: LuccFineTuningDatasetArgs,
                 dataloader_args: BaseDataloaderArgs,
                 batch_size: int = None):
        metadata_path = os.path.join(dataset_args.folder, self.metadata_filename)
        super().__init__(dataset_args, metadata_path)
        self.dataset_args = dataset_args
        self.dataloader_args = dataloader_args

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.image_arrays = None
        self.batch_size = batch_size

        # must save all hyperparameters for checkpoint
        self.save_hyperparameters()

    def make_dataset_args(self, sub_folder_name: str):
        sub_folder = os.path.join(self.dataset_args.folder, sub_folder_name)
        dataset_args = dataclasses.replace(self.dataset_args, folder=sub_folder)
        return dataset_args

    def setup(self, stage: [str] = None):
        if stage == "fit":
            self.train_dataset = LuccFineTuningDataset(self.make_dataset_args("train"))
            self.val_dataset = LuccFineTuningDataset(self.make_dataset_args("val"))
        if stage == "test":
            self.test_dataset = LuccFineTuningDataset(self.make_dataset_args("test"))
        if stage == "predict":
            predict_args = LuccPredictDatasetArgs(image_arrays=self.image_arrays,
                                                  bands=self.dataset_args.bands,
                                                  image_size=self.dataset_args.image_size,
                                                  model_patch_size=self.dataset_args.model_patch_size,
                                                  norm_min=self.dataset_args.norm_min,
                                                  norm_max=self.dataset_args.norm_max)
            self.predict_dataset = LuccPredictDataset(predict_args)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **dataclasses.asdict(self.dataloader_args))

    def val_dataloader(self):
        dataloader_args = dataclasses.replace(self.dataloader_args, shuffle=False)
        return DataLoader(self.val_dataset, **dataclasses.asdict(dataloader_args))

    def test_dataloader(self):
        dataloader_args = dataclasses.replace(self.dataloader_args, shuffle=False)
        return DataLoader(self.test_dataset, **dataclasses.asdict(dataloader_args))

    def predict_dataloader(self):
        dataloader_args = dataclasses.replace(self.dataloader_args, shuffle=False, batch_size=self.batch_size)
        return DataLoader(self.predict_dataset, **dataclasses.asdict(dataloader_args))
