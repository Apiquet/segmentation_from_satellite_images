"""MiniFrance dataset implementation."""

import copy
from abc import abstractmethod
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AbstractDataset(Dataset):
    """Abstract class for a custom Dataset, an example can be found under data/datasets/mini_france/mini_france_dataset.py."""

    features_preprocess: list[Callable]

    @abstractmethod
    def orderly_take(self, indexes: list[int]) -> None:
        """Remove all elements not contained in 'indexes' and apply their order."""
        raise NotImplementedError("orderly_take not implemented.")

    @abstractmethod
    def get_data_info_at_index(self, index: int) -> dict[str, Path]:
        """Get info about the data at specific index."""
        raise NotImplementedError("get_data_info_at_index not implemented.")

    @abstractmethod
    def get_mean_std_per_channel_from_features(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Read all element and compute per channel mean and std.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: mean and std values, shape: (N samples,)
        """
        raise NotImplementedError("get_mean_std_per_channel_from_features not implemented.")


def get_orderly_filtered_dataset(dataset: AbstractDataset, indexes: list[int]) -> AbstractDataset:
    """Return a copy of the dataset with elements contained at indexes with the same order."""
    filtered_dataset = copy.deepcopy(dataset)
    filtered_dataset.orderly_take(indexes)
    return filtered_dataset


def get_train_val_splits(dataset: AbstractDataset, train_ratio: float, seed: Optional[int] = 42) -> tuple[AbstractDataset, AbstractDataset]:
    """Get a training and a valid dataloaders from a given dataset.

    Args:
        dataset: Tensorflow dataset.
        train_ratio (float): ratio of the train set (train set = train_ratio*dataset)
        seed: seed value to get a reproducible random number. Can be set to None to disable.

    Raises:
        ValueError: If not 0 < train_ratio < 1

    Returns:
        train_dataloader, val_dataloader, train_dataset, val_dataset
    """
    if not 0 < train_ratio < 1.0:
        raise ValueError(
            f"Train ratio must be between 0 and 1 excluded, got {train_ratio}.",
        )

    generator = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
    if seed is not None:
        np.random.seed(seed)

    train_subset, valid_subset = random_split(dataset, [train_ratio, round(1.0 - train_ratio, 2)], generator=generator)
    train_ds = get_orderly_filtered_dataset(dataset, train_subset.indices)
    valid_ds = get_orderly_filtered_dataset(dataset, valid_subset.indices)

    return train_ds, valid_ds
