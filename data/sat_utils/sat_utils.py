"""Functions to download satellite images."""
from pathlib import Path

import numpy as np
import rasterio
from geopandas import GeoSeries
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject
from skimage.transform import resize


def merge_and_reproject_features_labels(features_to_merge: list[Path], merged_feature_path: Path, label_crs: str, label_geo_series: GeoSeries) -> None:
    """Merge all the given features to one raster reprojected according to the label file.

    Args:
        features_to_merge (list[Path]): list of the features to merge
        merged_feature_path (Path): path to the final merged feature
        label_crs (str): crs of the label to reproject features on
        label_geo_series (GeoSeries): geo serie to crop features
    """
    for feature_path in features_to_merge:
        # reproject feature in 4326 to label crs
        with rasterio.open(feature_path) as features_4326:
            t_transform, t_width, t_height = calculate_default_transform(features_4326.crs, label_crs, features_4326.width, features_4326.height, *features_4326.bounds)
            with rasterio.open(
                feature_path, "w", driver="GTiff", height=t_height, width=t_width, count=features_4326.count, dtype=features_4326.meta["dtype"], crs=label_crs, transform=t_transform
            ) as dst:
                reproject(
                    source=features_4326.read(), destination=rasterio.band(dst, 1), src_transform=features_4326.transform, src_crs=features_4326.crs, dst_transform=dst.transform, dst_crs=dst.crs
                )

        # crop reprojected feature to the label shape
        with rasterio.open(feature_path) as features_reprojected_raster:
            out_image, out_transform = mask(features_reprojected_raster, label_geo_series.geometry, crop=True)
            out_meta = features_reprojected_raster.meta.copy()

        out_meta.update({"driver": "Gtiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})

        with rasterio.open(feature_path, "w", **out_meta) as dst:
            dst.write(out_image)

    # merge all features
    features = []
    features_names = []
    for feature_idx, feature_path in enumerate(features_to_merge):
        with rasterio.open(feature_path) as feature_raster:
            if feature_idx == 0:
                height, width = feature_raster.height, feature_raster.width
                feature_meta = feature_raster.meta.copy()
            features_names.append(feature_path.stem)
            features.append(resize(feature_raster.read(), (1, height, width), order=0))
            feature_path.unlink()
    feature_meta.update(count=len(features_to_merge))
    with rasterio.open(merged_feature_path, "w", **feature_meta) as dst:
        dst.write(np.concatenate(features))
        dst.descriptions = tuple(features_names)
