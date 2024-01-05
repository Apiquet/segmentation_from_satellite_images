"""MiniFrance dataset implementation."""
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import rasterio as rio
import torch
from skimage.transform import resize
from torchvision.transforms import ToTensor
from tqdm import tqdm

from data.datasets.datasets_utils import AbstractDataset
from data.sat_utils.sat_download import download_s1_vh_vv_features

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MiniFranceDataset(AbstractDataset):
    """Dataset to load MiniFrance data."""

    def __init__(
        self,
        db_path: Path,
        train_ratio: float,
        gee_project_name: Optional[str] = None,
        keep_n_elements: Optional[int] = None,
        features_preprocess: Optional[list[Callable]] = None,
        labels_preprocess: Optional[list[Callable]] = None,
    ) -> None:
        """Init class for Mini France reader.

        Args:
            db_path (Path): path to the folder containing features and labels folder with tif files
            train_ratio (float): ratio for the training set
            gee_project_name (Optional[str]): name of the google earth engine project
            keep_n_elements (Optional[int], optional): To specifiy to keep only N elements, useful for testing. None to disable. Defaults to None.
            features_preprocess (Optional[list[Callable]], optional): preprocessing to apply on features. Defaults to None.
            labels_preprocess (Optional[list[Callable]], optional): preprocessing to apply on labels. Defaults to None.
        """
        super().__init__()
        self.features_preprocess = []
        self.labels_preprocess = []
        self.db_path = db_path

        self.data: dict[int, dict[str, Path]] = {}

        self.rescaled_features_path = db_path / f"preprocessed_features_{train_ratio=}"
        self.rescaled_labels_path = db_path / f"preprocessed_labels_{train_ratio=}"

        if not (self.rescaled_features_path / "mean_std_per_band.json").is_file():
            feature_paths = sorted((db_path / "features").rglob("*.tif"))
            label_paths = sorted((db_path / "labels").rglob("*.tif"))

            # get features
            if len(feature_paths) < len(label_paths):
                if gee_project_name is None:
                    raise ValueError("gee_project_name should not be None if sat images have to be downloaded.")
                download_s1_vh_vv_features(db_path=db_path, gee_project_name=gee_project_name)
                feature_paths = sorted((db_path / "features").rglob("*.tif"))
                label_paths = sorted((db_path / "labels").rglob("*.tif"))

            # store data paths
            for sample_idx, (feature_path, label_path) in enumerate(zip(feature_paths, label_paths)):
                if feature_path.stem != label_path.stem:
                    raise ValueError(f"{feature_path} does not match {label_path}.")
                self.data[sample_idx] = {"feature_path": feature_path, "label_path": label_path}

            # get min height and min width to rescale features/labels to same resolution
            min_height, min_width = self.get_min_height_width_per_channel_feature()
            if min_height % 2 != 0:
                min_height -= 1
            if min_width % 2 != 0:
                min_width -= 1
            self.features_preprocess = [lambda x: resize(x, (min_height, min_width), order=0)]
            self.labels_preprocess = [lambda x: resize(x, (min_height, min_width), order=0)]

            # save rescaled raw features and labels and a json file into rescaled_features_path containing mean and std per band
            self.save_features_labels_and_normalization_info_to_folders()

        self.features_preprocess = features_preprocess if isinstance(features_preprocess, list) else []
        self.labels_preprocess = labels_preprocess if isinstance(labels_preprocess, list) else []

        feature_paths = sorted((self.rescaled_features_path).rglob("*.pt"))
        label_paths = sorted((self.rescaled_labels_path).rglob("*.pt"))

        for sample_idx, (feature_path, label_path) in enumerate(zip(feature_paths, label_paths)):
            if feature_path.stem != label_path.stem:
                raise ValueError(f"{feature_path} does not match {label_path}.")
            self.data[sample_idx] = {"feature_path": feature_path, "label_path": label_path}

        if keep_n_elements is not None:
            self.data = dict(list(self.data.items())[:keep_n_elements])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get preprocessed (feature, label) at a given index."""
        if self.data[index]["feature_path"].suffix == ".tif":
            features = rio.open(self.data[index]["feature_path"]).read().transpose(1, 2, 0).astype(float)
        elif self.data[index]["feature_path"].suffix == ".pt":
            features = torch.load(self.data[index]["feature_path"])

        for preprocess in self.features_preprocess:
            features = preprocess(features)
        if not isinstance(features, torch.Tensor):
            features = ToTensor()(features)
        features = features.to(DEVICE).float()

        if self.data[index]["label_path"].suffix == ".tif":
            labels = rio.open(self.data[index]["label_path"]).read().transpose(1, 2, 0).astype(float)
        elif self.data[index]["label_path"].suffix == ".pt":
            labels = torch.load(self.data[index]["label_path"])

        for preprocess in self.labels_preprocess:
            labels = preprocess(labels)
        if not isinstance(labels, torch.Tensor):
            labels = ToTensor()(labels).squeeze()
        labels = labels.to(DEVICE).long()
        return features, labels

    def get_data_info_at_index(self, index: int) -> dict[str, Path]:
        """Get info about the data at specific index."""
        return {"feature_path": self.data[index]["feature_path"], "label_path": self.data[index]["label_path"]}

    def get_mean_std_per_channel_from_features(self, features: torch.Tensor) -> tuple[dict[int, dict[str, float]]]:
        """Read all element and compute per channel mean and std.

        Args:
            features (torch.Tensor): features to compute mean and std from, shape: (N samples, C channels, W, H)

        Returns:
            tuple[dict[int, dict[str, float]]]: dict with the band indexes as keys and values are:
                "mean" key with the mean value
                "std" key with the std value
        """
        mean_per_band = torch.mean(features, axis=(0, 2, 3))
        std_per_band = torch.std(features, axis=(0, 2, 3))
        mean_std_per_band = {}
        for band_idx, (band_mean, band_std) in enumerate(zip(mean_per_band, std_per_band)):
            mean_std_per_band[band_idx] = {"mean": band_mean.cpu().detach().item(), "std": band_std.cpu().detach().item()}
        return mean_std_per_band

    def get_min_height_width_per_channel_feature(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Read all element to get the minimum width and height to resize the data.

        Returns:
            tuple[torch.Tensor,torch.Tensor]: min height and min width values
        """
        min_height, min_width = np.inf, np.inf
        for data_idx in tqdm(range(len(self.data)), desc="Calculating min height and min width for rescaling"):
            height, width = self[data_idx][0].shape[1:]
            min_height = min(min_height, height)
            min_width = min(min_width, width)
        return min_height, min_width

    def save_features_labels_and_normalization_info_to_folders(self) -> None:
        """Read all element to save tensors to self.rescaled_features_path and self.rescaled_labels_path folders."""
        features_list = []

        for data_idx in tqdm(range(len(self)), desc="Save features and labels", total=len(self)):
            feature_path = self.get_data_info_at_index(data_idx)["feature_path"]
            preprocessed_feature_folder = self.rescaled_features_path / feature_path.relative_to(self.db_path / "features").parent
            preprocessed_feature_folder.mkdir(parents=True, exist_ok=True)
            preprocessed_feature_path = preprocessed_feature_folder / f"{feature_path.stem}.pt"

            label_path = self.get_data_info_at_index(data_idx)["label_path"]
            preprocessed_label_folder = self.rescaled_labels_path / label_path.relative_to(self.db_path / "labels").parent
            preprocessed_label_folder.mkdir(parents=True, exist_ok=True)
            preprocessed_label_path = preprocessed_label_folder / f"{label_path.stem}.pt"

            feature, label = self[data_idx]

            features_list.append(feature)

            if not preprocessed_feature_path.is_file():
                torch.save(feature, preprocessed_feature_path)

            if not preprocessed_label_path.is_file():
                torch.save(label, preprocessed_label_path)

        mean_std_per_band = self.get_mean_std_per_channel_from_features(features=torch.stack(features_list, dim=0))

        # save normalization data to json file
        with open(self.rescaled_features_path / "mean_std_per_band.json", "w") as json_file:
            json.dump(mean_std_per_band, json_file, indent=4)

    def orderly_take(self, indexes: list[int]) -> None:
        """Remove all elements not contained in 'indexes' and apply their order."""
        filtered_data = {}
        for new_index, index_to_extract in enumerate(indexes):
            filtered_data[new_index] = self.data[index_to_extract]
        self.data = filtered_data

    def __len__(self) -> int:
        """Get the total size of the dataset."""
        return len(self.data)
