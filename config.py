"""Class to setup a training."""

from pathlib import Path
from typing import Callable, Optional

from torch import cuda, device
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy

from data.datasets.datasets_utils import get_train_val_splits
from data.datasets.mini_france.mini_france_dataset import MiniFranceDataset
from data.datasets.mini_france.mini_france_utils import REMAP_LABELS
from data.preprocessing.labels_preprocessing import remap_labels
from models.unet import UNet
from viewers.viewers import MiniFranceSamplesViewer


class Config:
    """Variables needed to run a training."""

    def __init__(self, db_path: Path, gee_project_name: Optional[str] = None) -> None:
        """Init all training variables."""
        # Training vars
        self.batch_size = 8
        self.epochs_count = 1
        self.loss_func = CrossEntropyLoss(ignore_index=0)

        # Dataset variables
        self.keep_n_elements = 80
        self.train_ratio = 0.5
        self.db_path = db_path
        self.features_preprocess = None
        self.labels_preprocess = [lambda x: remap_labels(x, remap_names=REMAP_LABELS)]

        # Model variables
        self.number_of_classes = len(REMAP_LABELS)
        self.model = UNet(in_channels=2, n_classes=self.number_of_classes, padding=True)
        self.model.to(device("cuda" if cuda.is_available() else "cpu"))

        # Optimizer variables
        self.init_lr = 0.001
        self.optimizer = Adam(self.model.parameters(), lr=self.init_lr)

        # Dataset
        dataset = MiniFranceDataset(
            db_path=self.db_path, keep_n_elements=self.keep_n_elements, features_preprocess=self.features_preprocess, labels_preprocess=self.labels_preprocess, gee_project_name=gee_project_name
        )
        self.train_dl, self.val_dl, self.train_ds, self.val_ds = get_train_val_splits(dataset=dataset, batch_size=self.batch_size, train_ratio=self.train_ratio)

        # Metrics
        self.metrics: list[Metric] = [MulticlassAccuracy(self.number_of_classes)]
        self.viewers: list[Callable] = [MiniFranceSamplesViewer(samples_count_to_visualize=15, dataset=self.val_ds, batch_size=self.batch_size)]
