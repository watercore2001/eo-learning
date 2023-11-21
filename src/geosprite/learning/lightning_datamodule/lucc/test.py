import numpy as np
from geosprite.learning.lightning_datamodule.lucc.dataset import *



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = LuccPredictDataset(LuccFineTuningDatasetArgs(folder="/mnt/data/dataset/water_predict", image_size=512,
                                                           model_patch_size=4, use_norm=True))
    dataloader = DataLoader(dataset=dataset, batch_size=4)
    for batch in dataloader:
        pass
