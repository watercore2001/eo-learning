from geosprite.learning.lightning_datamodule.pastis.dataset_for_predict import PastisDataset, DatasetArgs
from geosprite.learning.lightning_module import ClassificationModule
from torchmetrics.classification import Accuracy
import argparse
import torch
import os
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt_path", type=str)
    parser.add_argument("-f", "--pastis_folder", type=str)
    parser.add_argument("-o", "--output_folder", type=str)
    parser.add_argument("-d", "--device", type=str)
    return parser.parse_args()


def eval_dataset(dataloader: DataLoader, model: nn.Module, pred_folder: str, stage: str, device: str):
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    metric = Accuracy(task="multiclass", average="none", num_classes=20, ignore_index=19)
    with torch.no_grad():
        for x, y, patch_id in dataloader:
            x = [i.to(device) for i in x]
            y = y.to(device)
            pred = model(x)
            metric.update(pred, y)
            pred_path = os.path.join(pred_folder, f"{patch_id.numpy()[0]}.npy")
            pred_data = pred.argmax(1).cpu().numpy()[0]

            # np.save(pred_path, pred_data)

        print(metric.compute())
        # np.save(os.path.join(pred_folder, f"{stage}_matrix.npy"), matrix)
        # save_confusion_matrix_image(matrix, save_filepath=os.path.join(pred_folder, f"{stage}_matrix.png"))


def save_confusion_matrix_image(confusion_matrix, pred_folder: str, stage: str):
    # labels = ["背景", "牧场", "冬软粒小麦", "玉米", "冬大麦", "冬油菜籽", "春大麦", "向日葵", "葡萄藤", "甜菜", "冬小黑麦", "冬硬粒小麦", "水果", "土豆", "豆科饲料", "大豆", "果园", "谷物", "高粱", "空标签"]
    labels = [i for i in range(20)]
    fig, ax = plt.subplots(figsize=(15, 15), dpi=100)
    sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted labels', fontsize=20)
    ax.set_ylabel('True labels', fontsize=20)
    ax.set_title(f'{stage} Confusion Matrix', fontsize=25)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.savefig(os.path.join(pred_folder, f"{stage.lower()}_confusion_matrix.png"))


def main():
    args = parse_args()
    model = ClassificationModule.load_from_checkpoint(args.ckpt_path)
    train_dataset = PastisDataset(DatasetArgs(folder=args.pastis_folder, folds=[1, 2, 3]))
    train_dataloader = DataLoader(train_dataset)
    # val_dataset = PastisDataset(DatasetArgs(folder=args.pastis_folder, folds=[4]))
    # val_dataloader = DataLoader(val_dataset)
    # test_dataset = PastisDataset(DatasetArgs(folder=args.pastis_folder, folds=[5]))
    # test_dataloader = DataLoader(test_dataset)
    eval_dataset(train_dataloader, model, args.output_folder, stage="train", device=args.device)
    # eval_dataset(val_dataloader, model, args.output_folder, stage="val", device=args.device)
    # eval_dataset(test_dataloader, model, args.output_folder, stage="test", device=args.device)


if __name__ == "__main__":
    main()
