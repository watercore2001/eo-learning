import copy
import os
import logging
import dataclasses
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from .transform import ToNormalizedTensor
from .dataset import RasterFolderDataset, DatesetArgs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclasses.dataclass
class DataloaderArgs:
    batch_size: int
    num_workers: int
    pin_memory: bool


class RasterFolderModule(LightningDataModule):
    def __init__(
            self,
            root: str,
            dataset_args: DatesetArgs,
            dataloader_args: DataloaderArgs
    ):
        super().__init__()

        self.root = root
        self.dataset_args = dataset_args
        self.dataloader_args = dataloader_args

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # save hparams into self.hparams
        self.save_hyperparameters()

    def make_dataset(self, root: str) -> RasterFolderDataset:
        updated_args = {"image_folder": os.path.join(root, self.dataset_args.image_folder),
                        "gt_folder": os.path.join(root, self.dataset_args.gt_folder)}
        dataset_args = dataclasses.replace(copy.deepcopy(self.dataset_args), **updated_args)
        # update image and gt folder
        dataset = RasterFolderDataset(dataset_args)
        assert len(dataset) > 0, f"Dataset folder is empty. " \
                                 f"image folder is {dataset_args.image_folder}. " \
                                 f"gt folder is '{dataset_args.gt_folder}."

        return dataset

    def setup(self, stage: str = None):

        if stage == 'fit':
            train_root = os.path.join(self.root, "train")
            self.train_dataset = self.make_dataset(train_root)

            if isinstance(self.train_dataset.image_transform, ToNormalizedTensor):
                logger.info("calculating mean and std for Transform 'ToNormalizedTensor' ...")

                self.train_dataset.image_transform.init_mean_std(self.train_dataset)
                # 1. it will be used in val and test
                self.dataset_args.image_transform = self.train_dataset.image_transform
                # 2. it will be used in checkpoint
                assert self.hparams.dataset_args is self.dataset_args

                logger.info("calculating mean and std for Transform 'ToNormalizedTensor' ... Done")

            val_root = os.path.join(self.root, "val")
            self.val_dataset = self.make_dataset(val_root)

        elif stage == 'validate':
            val_root = os.path.join(self.root, "val")
            self.val_dataset = self.make_dataset(val_root)

        elif stage == 'test':
            test_root = os.path.join(self.root, "test")
            self.val_dataset = self.make_dataset(test_root)

    def train_dataloader(self) -> DataLoader:

        return DataLoader(
            self.train_dataset,
            **dataclasses.asdict(self.dataloader_args)
        )

    def val_dataloader(self) -> DataLoader:

        return DataLoader(
            self.val_dataset,
            **dataclasses.asdict(self.dataloader_args)
        )

    def test_dataloader(self) -> DataLoader:

        return DataLoader(
            self.test_dataset,
            **dataclasses.asdict(self.dataloader_args)
        )
