"""Function to preprocess features."""

import numpy as np

from data.datasets.mini_france.mini_france_utils import RAW_CLASS_NAME_TO_LABEL


def remap_labels(labels: np.ndarray, remap_names: dict[int, list[str]]) -> np.ndarray:
    """Modify labels according to remap_names.

    Args:
        labels (np.ndarray): labels to remap
        remap_names (dict[int, list[str]]):
            key: the new label value,
            value: the list of the old classes to merge

    Returns:
        np.ndarray: remapped labels
    """
    for new_idx, (_, old_class_names) in enumerate(remap_names.items()):
        for old_class_name in old_class_names:
            labels[labels == RAW_CLASS_NAME_TO_LABEL[old_class_name]] = new_idx
    return labels
