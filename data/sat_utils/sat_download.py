"""Functions to download satellite images."""
import re
from pathlib import Path
from shutil import move

import ee
import geemap
import numpy as np
import rasterio
from geopandas import GeoSeries
from shapely.geometry import box
from tqdm import tqdm

from data.sat_utils.sat_s2_utils import add_cld_shdw_mask, apply_cld_shdw_mask, get_s2_sr_cld_col
from data.sat_utils.sat_utils import merge_and_reproject_features_labels


def download_s1_s2_features(db_path: Path, gee_project_name: str, unique_orbit_sens: bool = True, unique_orbit_number: bool = True) -> None:
    """Download multi-temporal VH and VV S1 data as a yearly averaged image and S2 optical bands.

    It will save one feature per label tif file, under the features folder, with 14 channels: S2 bands and S1 VH and VV. See FEATURES_NAMES_TO_BAND_IDX from mini_france_utils.py.

    Args:
        db_path (Path): path to the database containing labels folder
        gee_project_name (str): name of the google earth engine project
        unique_orbit_sens (str): to filter with the more frequent orbit sens (ASCENDING or DESCENDING). Defaults to True.
        unique_orbit_number (bool, optional): to filter with the more frequent orbit number. Defaults to True.
    """
    # inititalize google earth engine
    ee.Initialize(project=gee_project_name)

    label_paths = sorted(list((db_path / "labels").rglob("*.tif")))
    for label_path in tqdm(label_paths, total=len(label_paths), desc="Downloading features"):
        # get feature path to save
        feature_folder = db_path / label_path.relative_to(db_path).parent.as_posix().replace("labels", "features")
        feature_folder.mkdir(parents=True, exist_ok=True)
        feature_path = feature_folder / label_path.name
        if feature_path.is_file():
            continue

        # get date of the annotation
        year_pattern = re.compile(r"\d+\d+-([0-9][0-9][0-9][0-9])-", re.IGNORECASE).search(label_path.stem)
        if year_pattern is None:
            raise ValueError(f"Year pattern not found in {label_path}")
        year = max(2014, int(year_pattern.group(1)))
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        # read tif file to extract coordinates
        with rasterio.open(label_path) as label_raster:
            box_raster = box(label_raster.bounds.left, label_raster.bounds.bottom, label_raster.bounds.right, label_raster.bounds.top)
            geo_series = GeoSeries([box_raster], crs=label_raster.crs)
        coordinates = list(geo_series.to_crs("EPSG:4326").unary_union.envelope.exterior.coords)
        geom = ee.Geometry.Polygon([coordinates])
        feature = ee.Feature(geom, {})
        aoi = feature.geometry()

        # get S1 polarisations of the specific region
        dataset = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi)
        images_collection = dataset.filter(ee.Filter.eq("instrumentMode", "IW")).filterDate(start_date, end_date)

        if unique_orbit_sens:
            # filter the results with the most present orbit sens
            orbit_sens_list, counts = np.unique(
                images_collection.aggregate_array("orbitProperties_pass").getInfo(),
                return_counts=True,
            )
            indices = np.argsort(counts)
            orbit_sens = orbit_sens_list[indices[-1]]
            images_collection = images_collection.filter(ee.Filter.eq("orbitProperties_pass", orbit_sens))

        if unique_orbit_number:
            # filter the results with the most present orbit number
            orbit_numbers, counts = np.unique(
                images_collection.aggregate_array("relativeOrbitNumber_start").getInfo(),
                return_counts=True,
            )
            indices = np.argsort(counts)
            orbit_number = int(orbit_numbers[indices[-1]])
            images_collection = images_collection.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit_number))

        # get VH and VV polarisations
        vh_collection = images_collection.select("VH").filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        vv_collection = images_collection.select("VV").filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        vh_image = ee.Image(vh_collection.mean())
        vv_image = ee.Image(vv_collection.mean())

        # save both polarisations
        vh_temp = feature_path.parent / f"{feature_path.stem}_vh.tif"
        vv_temp = feature_path.parent / f"{feature_path.stem}_vv.tif"
        if not vh_temp.is_file():
            geemap.ee_export_image(vh_image, filename=vh_temp, scale=15, region=aoi, file_per_band=False)
        if not vv_temp.is_file():
            geemap.ee_export_image(vv_image, filename=vv_temp, scale=15, region=aoi, file_per_band=False)

        # get S2 image of the specified region
        year = max(2017, year)
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        dataset = get_s2_sr_cld_col(aoi=aoi, start_date=start_date, end_date=end_date)
        s2_sr = dataset.map(add_cld_shdw_mask).map(apply_cld_shdw_mask)
        s2_sr_median = s2_sr.median()

        s2_temp = feature_path.parent / f"{feature_path.stem}_s2.tif"
        if len(list(s2_temp.parent.glob(f"{s2_temp.stem}*.tif"))) == 0:
            geemap.ee_export_image(s2_sr_median, filename=s2_temp, scale=15, region=aoi, file_per_band=True)

        downloading_files = sorted(list(feature_path.parent.rglob(f"{feature_path.stem}*")))

        # move labels to another folder if downloading failed
        if len(downloading_files) < 14:
            unusable_labels_path = db_path / label_path.relative_to(db_path).parent.as_posix().replace("labels", "unusable_labels")
            unusable_labels_path.mkdir(parents=True, exist_ok=True)
            unusable_label_path = unusable_labels_path / label_path.name
            move(label_path, unusable_label_path)
            for downloading_file in downloading_files:
                downloading_file.unlink()
            continue

        merge_and_reproject_features_labels(features_to_merge=downloading_files, merged_feature_path=feature_path, label_crs=label_raster.crs, label_geo_series=geo_series)
