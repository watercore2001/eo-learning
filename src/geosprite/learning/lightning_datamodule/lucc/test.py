import numpy as np
from geosprite.learning.lightning_datamodule.lucc.dataset import *

dataset = LuccFineTuningDataset(FineTuningDatasetArgs(use_norm=True, folder="/mnt/dataset/building_stack/train"))
for data in dataset:
    pass
