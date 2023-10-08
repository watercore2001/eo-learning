# Copyright (c) GeoSprite. All rights reserved.
#
# Author: Jia Song
#
import torch
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
import os


# use for config file

class LoggerLightningCLI(LightningCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_class_arguments(WandbLogger, "wandb_logger")
        parser.link_arguments("trainer.default_root_dir", "wandb_logger.save_dir")
        parser.set_defaults(
            {
                "wandb_logger.project": self.model_class.__name__,
                "wandb_logger.name": datetime.now().strftime('%Y.%m.%d.%H%M'),
                "wandb_logger.version": datetime.now().strftime('%Y.%m.%d.%H%M')
            }
        )

    def instantiate_trainer(self, **kwargs) -> Trainer:
        # check and customize trainer logger with tensorboard_logger config

        subcommand = self.config_init["subcommand"]
        config = self.config_init[subcommand]

        wandb_logger = config["wandb_logger"]
        trainer_logger = config["trainer"]["logger"]

        # save all parameter in workspace
        os.makedirs(os.path.join(wandb_logger.save_dir, wandb_logger.name, wandb_logger.version), exist_ok=True)
        self.save_config_kwargs.update({"config_filename": f"{wandb_logger.name}/{wandb_logger.version}/config.yaml"})

        # trainer_logger's default value is True
        if wandb_logger is not None and trainer_logger is True:
            config["trainer"]["logger"] = wandb_logger

        return super().instantiate_trainer()

    def after_fit(self):
        self.trainer.test(model=self.model, datamodule=self.datamodule, ckpt_path="best")


def main():
    # lower precision for higher speed
    torch.set_float32_matmul_precision("medium")

    LoggerLightningCLI(
        LightningModule,
        LightningDataModule,
        save_config_kwargs={"overwrite": True},
        subclass_mode_model=True,
        subclass_mode_data=True
    )


if __name__ == '__main__':
    main()
