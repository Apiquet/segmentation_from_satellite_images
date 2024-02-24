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

    def __init__(self, samples_count_to_visualize: int, dataset: Dataset, batch_size: int, classes_to_rgb: dict[str, tuple[int, int, int]], display_prediction: bool = True) -> None:
        """Constructor for MiniFranceSamplesViewer.

        Args:
            samples_count_to_visualize (int): number of samples to visualize in the final figure
            dataset (Dataset): dataset to get data info at specific indexes
            batch_size (int): size of each batch
            classes_to_rgb (dict[str, tuple[int, int, int]]): color and name to assign to each label indexe in order (first class to label==0, etc.)
            display_prediction (bool, optional): whether to display the predictions in the final viz. Defaults to True.
        """
        self.display_prediction = display_prediction
        self.samples_count_to_visualize = samples_count_to_visualize
        self.dataset = dataset
        self.current_batch_index = -1
        self.batch_size = batch_size
        self.classes_to_rgb = classes_to_rgb
        self.samples_indexes_to_plot = np.random.choice(len(dataset), self.samples_count_to_visualize, replace=False)
        self.features = []
        self.labels = []
        self.preds = []
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
        for el_idx, (label, pred) in enumerate(zip(labels, preds)):
            sample_idx_in_dataset = current_samples_indexes[el_idx]
            if sample_idx_in_dataset in self.samples_indexes_to_plot:
                self.features.append(self.dataset[sample_idx_in_dataset][0])
                self.labels.append(label)
                self.preds.append(pred)
                self.data_info.append(self.dataset.get_data_info_at_index(sample_idx_in_dataset))

    def compute(self) -> Figure:
        """Compute viewer to get final figure."""
        # set figure
        cols = ["Feature RGB\n", "Feature VH\n", "Feature VV\n", "Prediction\n", "Label\n", "Legend:\n"]
        if not self.display_prediction:
            cols.remove("Prediction\n")
        cols_count = len(cols)
        fig, axarr = plt.subplots(self.samples_count_to_visualize, cols_count, figsize=(14, 12 * (self.samples_count_to_visualize / cols_count)))
        if self.samples_count_to_visualize == 1:
            axarr = np.array([axarr])
        plt.suptitle(str(self), fontsize=20)

        labels_names = list(self.classes_to_rgb.keys())
        labels_colors = list(self.classes_to_rgb.values())

        # set column names
        for ax, col in zip(axarr[0], cols):
            ax.set_title(col, fontweight="bold", fontsize=14)

        cmap = colors.ListedColormap(np.array(list(self.classes_to_rgb.values())) / 255)
        for sample_idx, (ax, data_info, feature, label, pred) in enumerate(zip(axarr, self.data_info, self.features, self.labels, self.preds)):
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

            if self.display_prediction:
                final_pred = torch.argmax(pred, dim=0)
                ax[3].imshow(final_pred.detach().cpu().numpy(), cmap=cmap)
                ax[3].set_axis_off()

            ax[3 + int(self.display_prediction)].imshow(label.int().detach().cpu().numpy(), cmap=cmap)
            ax[3 + int(self.display_prediction)].set_axis_off()

            ax[4 + int(self.display_prediction)].set_axis_off()
            if sample_idx == 0:
                blank_im = np.ones_like(label.int().detach().cpu().numpy())
                ax[4 + int(self.display_prediction)].imshow(blank_im, cmap="gray", vmin=0, vmax=1)
                for label_idx, (label_name, label_color) in enumerate(zip(labels_names, labels_colors)):
                    ax[4 + int(self.display_prediction)].text(
                        -0.15,
                        1.0 - 0.16 * label_idx,
                        f"- {label_name[:40]}{'...' if len(label_name) > 40 else ''}",
                        horizontalalignment="left",
                        color=np.array(label_color) / 255.0,
                        transform=ax[4 + int(self.display_prediction)].transAxes,
                    )

        return fig
