"""Function to preprocess features."""

import numpy as np


def normalization_per_channel(features: np.ndarray, mean_values: np.ndarray, std_values: np.ndarray) -> np.ndarray:
    """Normalize each feature channel: (feature-mean)/std.

    Args:
        features (np.ndarray): tensor format [C,H,W]
        mean_values (np.ndarray): C values
        std_values (np.ndarray): C values

    Returns:
        Tensor: normalized tensor
    """
    normalize_features = features - mean_values[None, None, :]
    normalize_features /= std_values[None, None, :]
    return normalize_features
