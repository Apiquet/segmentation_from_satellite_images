"""MiniFrance dataset implementation."""
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import rasterio as rio
import torch
from skimage.transform import resize
from torchvision.transforms import ToTensor
from tqdm import tqdm

from data.datasets.datasets_utils import AbstractDataset
from data.preprocessing.features_preprocessing import normalization_per_channel
from data.sat_utils.sat_download import download_s1_vh_vv_features

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MiniFranceDataset(AbstractDataset):
    """Dataset to load MiniFrance data."""

    def __init__(
        self,
        db_path: Path,
        gee_project_name: Optional[str] = None,
        keep_n_elements: Optional[int] = None,
        features_preprocess: Optional[list[Callable]] = None,
        labels_preprocess: Optional[list[Callable]] = None,
    ) -> None:
        """Init class for Mini France reader.

        Args:
            db_path (Path): path to the folder containing features and labels folder with tif files
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

        preprocessed_label_paths = sorted((db_path / "preprocessed_labels").rglob("*.pt"))
        label_paths = sorted((db_path / "labels").rglob("*.tif"))

        if (db_path / "labels").is_dir() and (len(preprocessed_label_paths) < len(label_paths) or len(label_paths) == 0):
            feature_paths = sorted((db_path / "features").rglob("*.tif"))
            label_paths = sorted((db_path / "labels").rglob("*.tif"))

            # get features
            if len(feature_paths) < len(label_paths) or len(label_paths) == 0:
                if gee_project_name is None:
                    raise ValueError("gee_project_name should not be None if sat images have to be downloaded.")
                download_s1_vh_vv_features(db_path=db_path, gee_project_name=gee_project_name)
                feature_paths = sorted((db_path / "features").rglob("*.tif"))
                label_paths = sorted((db_path / "labels").rglob("*.tif"))

            for sample_idx, (feature_path, label_path) in enumerate(zip(feature_paths, label_paths)):
                if feature_path.stem != label_path.stem:
                    raise ValueError(f"{feature_path} does not match {label_path}.")
                self.data[sample_idx] = {"feature_path": feature_path, "label_path": label_path}

            min_height, min_width = self.get_min_height_width_per_channel_feature()
            self.features_preprocess = [lambda x: resize(x, (min_height, min_width), order=0)]
            self.labels_preprocess = [lambda x: resize(x, (min_height, min_width), order=0)]

            mean_values, std_values = self.get_mean_std_per_channel_feature()
            self.features_preprocess.extend([lambda x: normalization_per_channel(x, mean_values=mean_values.cpu().numpy(), std_values=std_values.cpu().numpy())])

            self.save_features_labels_to_folders()

        self.features_preprocess = features_preprocess if isinstance(features_preprocess, list) else []
        self.labels_preprocess = labels_preprocess if isinstance(labels_preprocess, list) else []

        feature_paths = sorted((db_path / "preprocessed_features").rglob("*.pt"))
        label_paths = sorted((db_path / "preprocessed_labels").rglob("*.pt"))

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

    def get_mean_std_per_channel_feature(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Read all element and compute per channel mean and std.

        Returns:
            tuple[torch.Tensor,torch.Tensor]: mean and std values per channel, format: (number of channels,)
        """
        features_list = []
        for data_idx in tqdm(range(len(self.data)), desc="Calculating mean and std values per channel"):
            features_list.append(self[data_idx][0])
        features_list = torch.stack(features_list)
        return (torch.mean(features_list, axis=(0, 2, 3)), torch.std(features_list, axis=(0, 2, 3)))

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

    def save_features_labels_to_folders(self) -> None:
        """Read all element to save tensors to preprocessed_features and preprocessed_labels folders."""
        for data_idx in tqdm(range(len(self)), desc="Save features and labels", total=len(self)):
            feature_path = self.get_data_info_at_index(data_idx)["feature_path"]
            preprocessed_feature_folder = self.db_path / feature_path.relative_to(self.db_path).parent.as_posix().replace("features", "preprocessed_features")
            preprocessed_feature_folder.mkdir(parents=True, exist_ok=True)
            preprocessed_feature_path = preprocessed_feature_folder / f"{feature_path.stem}.pt"

            label_path = self.get_data_info_at_index(data_idx)["label_path"]
            preprocessed_label_folder = self.db_path / label_path.relative_to(self.db_path).parent.as_posix().replace("labels", "preprocessed_labels")
            preprocessed_label_folder.mkdir(parents=True, exist_ok=True)
            preprocessed_label_path = preprocessed_label_folder / f"{label_path.stem}.pt"

            if preprocessed_feature_path.is_file() and preprocessed_label_path.is_file():
                continue

            data = self[data_idx]

            if not preprocessed_feature_path.is_file():
                torch.save(data[0], preprocessed_feature_path)

            if not preprocessed_label_path.is_file():
                torch.save(data[1], preprocessed_label_path)

    def orderly_take(self, indexes: list[int]) -> None:
        """Remove all elements not contained in 'indexes' and apply their order."""
        filtered_data = {}
        for new_index, index_to_extract in enumerate(indexes):
            filtered_data[new_index] = self.data[index_to_extract]
        self.data = filtered_data

    def __len__(self) -> int:
        """Get the total size of the dataset."""
        return len(self.data)
