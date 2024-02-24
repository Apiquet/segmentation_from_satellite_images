"""Classes to create visualizations."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors
from matplotlib.figure import Figure
from PIL import Image, ImageEnhance
from torch import Tensor
from torch.utils.data.dataset import Dataset

from data.datasets.mini_france.mini_france_utils import FEATURES_NAMES_TO_BAND_IDX

warnings.filterwarnings("ignore")


class MiniFranceSamplesViewer:
    """Viewer to see VH VH and labels data."""

    def __init__(self, samples_count_to_visualize: int, dataset: Dataset, batch_size: int, classes_to_rgb: dict[str, tuple[int, int, int]]) -> None:
        """Constructor for MiniFranceSamplesViewer.

        Args:
            samples_count_to_visualize (int): number of samples to visualize in the final figure
            dataset (Dataset): dataset to get data info at specific indexes
            batch_size (int): size of each batch
            classes_to_rgb (dict[str, tuple[int, int, int]]): color and name to assign to each label indexe in order (first class to label==0, etc.)
        """
        self.samples_count_to_visualize = samples_count_to_visualize
        self.dataset = dataset
        self.current_batch_index = -1
        self.batch_size = batch_size
        self.classes_to_rgb = classes_to_rgb
        self.samples_indexes_to_plot = np.random.choice(len(dataset), self.samples_count_to_visualize, replace=False)
        self.features = []
        self.labels = []
        self.data_info = []

    def __str__(self):
        """Overload of the str conversion."""
        return f"MiniFrance Viewer with {self.samples_count_to_visualize} samples"

    def reset(self) -> None:
        """Reset internal variables."""
        self.features = []
        self.labels = []
        self.data_info = []
        self.current_batch_index = -1

    def update(self, preds: Tensor, labels: Tensor) -> None:
        """Update viewer with internal variables."""
        self.current_batch_index += 1
        if self.current_batch_index >= len(self.dataset):
            raise ValueError(f"Current batch index ({self.current_batch_index}) >= total db samples ({self.ds_size})")
        current_samples_indexes = [idx for idx in range(self.current_batch_index * self.batch_size, (self.current_batch_index + 1) * self.batch_size)]
        for el_idx, label in enumerate(labels):
            sample_idx_in_dataset = current_samples_indexes[el_idx]
            if sample_idx_in_dataset in self.samples_indexes_to_plot:
                self.features.append(self.dataset[sample_idx_in_dataset][0])
                self.labels.append(label)
                self.data_info.append(self.dataset.get_data_info_at_index(sample_idx_in_dataset))

    def compute(self) -> Figure:
        """Compute viewer to get final figure."""
        # set figure
        cols = ["Feature RGB\n\n", "Feature VH\n\n", "Feature VV\n\n", "Label\n\n"]
        cols_count = len(cols)
        fig, axarr = plt.subplots(self.samples_count_to_visualize, cols_count, figsize=(10, 10 * (self.samples_count_to_visualize / cols_count)))
        if self.samples_count_to_visualize == 1:
            axarr = np.array([axarr])
        plt.suptitle(str(self), fontsize=20)

        yticks_indexes = np.arange(len(self.classes_to_rgb))
        yticks_labels = list(self.classes_to_rgb.keys())
        yticks_colors = list(self.classes_to_rgb.values())

        # set column names
        for ax, col in zip(axarr[0], cols):
            ax.set_title(col, fontweight="bold", fontsize=14)

        cmap = colors.ListedColormap(np.array(list(self.classes_to_rgb.values())) / 255)
        for ax, data_info, feature, label in zip(axarr, self.data_info, self.features, self.labels):
            ax[0].set_title(data_info["feature_path"].stem, loc="left")

            # get RGB S2 representation as B4 B3 B2
            rgb = feature[[FEATURES_NAMES_TO_BAND_IDX["B4"], FEATURES_NAMES_TO_BAND_IDX["B3"], FEATURES_NAMES_TO_BAND_IDX["B2"]]].permute(1, 2, 0)

            # Rescale rgb image to [0, 1]
            rgb = (rgb - torch.min(rgb)) / (torch.max(rgb) - torch.min(rgb))

            # increase brightness
            img_enhancer = ImageEnhance.Brightness(Image.fromarray((rgb.detach().cpu().numpy() * 255.0).astype(np.uint8)))
            ax[0].imshow(img_enhancer.enhance(2.5))
            ax[0].set_axis_off()

            ax[1].imshow(feature[FEATURES_NAMES_TO_BAND_IDX["VH"]].detach().cpu().numpy(), cmap="Greys")
            ax[1].set_axis_off()

            ax[2].imshow(feature[FEATURES_NAMES_TO_BAND_IDX["VV"]].detach().cpu().numpy(), cmap="Greys")
            ax[2].set_axis_off()

            ax[3].imshow(label.int().detach().cpu().numpy(), cmap=cmap, extent=yticks_indexes)
            ax[3].get_xaxis().set_visible(False)
            ax[3].yaxis.set_label_position("right")
            ax[3].yaxis.tick_right()
            ax[3].set_yticklabels(yticks_labels)
            for y_idx, y_color in zip(yticks_indexes, yticks_colors):
                plt.setp(ax[3].get_yticklabels()[y_idx], color=np.array(y_color) / 255)
        fig.tight_layout(pad=2)
        return fig
