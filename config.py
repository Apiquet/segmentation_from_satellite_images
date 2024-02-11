"""Class to setup a training."""

from pathlib import Path
from typing import Callable, Optional

from torch import cuda, device
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy

from data.datasets.datasets_utils import get_train_val_splits
from data.datasets.mini_france.mini_france_dataset import MiniFranceDataset
from data.datasets.mini_france.mini_france_utils import REMAP_LABELS, REMAP_LABELS_TO_RGB
from data.preprocessing.features_preprocessing import normalization_per_channel
from data.preprocessing.labels_preprocessing import remap_labels
from models.unet import UNet
from viewers.viewers import MiniFranceSamplesViewer


class Config:
    """Variables needed to run a training."""

    def __init__(self, db_path: Path, gee_project_name: Optional[str] = None) -> None:
        """Init all training variables."""
        # Training vars
        self.batch_size = 16
        self.epochs_count = 100
        self.loss_func = CrossEntropyLoss(ignore_index=0)

        # Dataset variables
        self.keep_n_elements = None
        self.train_ratio = 0.8
        self.db_path = db_path
        self.tensors_width_height = (256, 256)  # features and labels width and height
        self.features_preprocess = None
        self.labels_preprocess = [lambda x: remap_labels(x, remap_names=REMAP_LABELS)]

        # Model variables
        self.number_of_classes = len(REMAP_LABELS)
        self.model = UNet(in_channels=14, n_classes=self.number_of_classes, padding=True)
        self.model.to(device("cuda" if cuda.is_available() else "cpu"))

        # Optimizer variables
        self.init_lr = 0.001
        self.optimizer = Adam(self.model.parameters(), lr=self.init_lr)

        # Dataset
        dataset = MiniFranceDataset(
            db_path=self.db_path,
            keep_n_elements=self.keep_n_elements,
            features_preprocess=self.features_preprocess,
            labels_preprocess=self.labels_preprocess,
            gee_project_name=gee_project_name,
            tensors_width_height=self.tensors_width_height,
        )

        # get train and validation sets
        self.train_ds, self.val_ds = get_train_val_splits(dataset=dataset, train_ratio=self.train_ratio)

        # compute mean and std values per channel on the train set
        mean_values, std_values = self.train_ds.get_mean_std_per_channel_from_features()

        # apply mean and std values as preprocessing for train and validation sets
        self.train_ds.features_preprocess.extend([lambda x: normalization_per_channel(x, mean_values=mean_values.cpu(), std_values=std_values.cpu())])
        self.val_ds.features_preprocess.extend([lambda x: normalization_per_channel(x, mean_values=mean_values.cpu(), std_values=std_values.cpu())])

        # create train and validation dataloader
        self.train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=False)
        self.val_dl = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)

        # Metrics
        self.metrics: list[Metric] = [MulticlassAccuracy(self.number_of_classes)]
        self.viewers: list[Callable] = [MiniFranceSamplesViewer(samples_count_to_visualize=5, dataset=self.val_ds, batch_size=self.batch_size, classes_to_rgb=REMAP_LABELS_TO_RGB)]
