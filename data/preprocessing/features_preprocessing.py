"""Function to preprocess features."""

from torch import Tensor


def normalization_per_channel(features: Tensor, mean_values: Tensor, std_values: Tensor) -> Tensor:
    """Normalize each feature channel: (feature-mean)/std.

    Args:
        features (Tensor): tensor format [C,H,W]
        mean_values (Tensor): C values
        std_values (Tensor): C values

    Returns:
        Tensor: normalized tensor
    """
    normalize_features = features - mean_values[:, None, None]
    normalize_features /= std_values[:, None, None]
    return normalize_features
