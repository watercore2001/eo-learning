from pytorch_lightning import LightningDataModule
import dataclasses

from torch.utils.data import DataLoader
from .dataset import DatasetArgs, LuccDataset
from ..base_dataloader import BaseDataloaderArgs

__all__ = ["DatasetArgs", "LuccDataset", "BaseDataloaderArgs", "LuccDataModule"]


class LuccDataModule(LightningDataModule):
    def __init__(self,
                 dataset_args: DatasetArgs,
                 dataloader_args: BaseDataloaderArgs):
        super().__init__()
        self.dataset_args = dataset_args
        self.dataloader_args = dataloader_args

        self.train_dataset = None

    def setup(self, stage: [str] = None) -> None:
        if stage == "fit":
            self.train_dataset = LuccDataset(self.dataset_args)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, **dataclasses.asdict(self.dataloader_args))
