"""Functions to download satellite images.

Functions from https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless.
"""
import ee

CLOUD_FILTER = 60
CLD_PRB_THRESH = 50
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 1
BUFFER = 50


def get_s2_sr_cld_col(aoi: ee.geometry, start_date: str, end_date: str) -> ee.ImageCollection:
    """Filter the SR and s2cloudless collections according to area of interest and date parameters.

    Also join them on the system:index property.
    The result is a copy of the SR collection where each image has a new 's2cloudless' property whose value is the corresponding s2cloudless image.

    Args:
        aoi (ee.geometry): area of interest
        start_date (str): start date format YYYY-MM-DD
        end_date (str): end date format YYYYY-MM-DD

    Returns:
        ee.ImageCollection: SR collection where each image has a new 's2cloudless' property whose value is the corresponding s2cloudless image
    """
    # Import and filter S2 SR.
    s2_sr_col = ee.ImageCollection("COPERNICUS/S2_SR").filterBounds(aoi).filterDate(start_date, end_date).filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", CLOUD_FILTER))

    # Import and filter s2cloudless.
    s2_cloudless_col = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterBounds(aoi).filterDate(start_date, end_date)

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(
        ee.Join.saveFirst("s2cloudless").apply(**{"primary": s2_sr_col, "secondary": s2_cloudless_col, "condition": ee.Filter.equals(**{"leftField": "system:index", "rightField": "system:index"})})
    )


def add_cloud_bands(img: ee.ImageCollection) -> ee.ImageCollection:
    """Add the s2cloudless probability layer and derived cloud mask as bands to an S2 SR image input.

    Args:
        img (ee.ImageCollection): _description_

    Returns:
        ee.ImageCollection: image with 2 bands added: s2cloudless probability and cloud mask bands
    """
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get("s2cloudless")).select("probability")

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename("clouds")

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))


def add_shadow_bands(img: ee.Image) -> ee.Image:
    """Add dark pixels, cloud projection, and identified shadows as bands to an S2 SR image input.

    Args:
        img (ee.Image): image from the above add_cloud_bands function

    Returns:
        ee.Image: image with bands dark pixels, cloud projection, and identified shadows
    """
    # Identify water pixels from the SCL band.
    not_water = img.select("SCL").neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select("B8").lt(NIR_DRK_THRESH * SR_BAND_SCALE).multiply(not_water).rename("dark_pixels")

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE")))

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (
        img.select("clouds")
        .directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST * 10)
        .reproject(**{"crs": img.select(0).projection(), "scale": 100})
        .select("distance")
        .mask()
        .rename("cloud_transform")
    )

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename("shadows")

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))


def add_cld_shdw_mask(img: ee.Image) -> ee.Image:
    """Assemble all of the cloud and cloud shadow components and produce the final mask.

    Args:
        img (ee.Image): image to add cloud and shadow masks

    Returns:
        ee.Image: image with band to inform is cloud and shadow
    """
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select("clouds").add(img_cloud_shadow.select("shadows")).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = is_cld_shdw.focalMin(2).focalMax(BUFFER * 2 / 20).reproject(**{"crs": img.select([0]).projection(), "scale": 20}).rename("cloudmask")

    # Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)


def apply_cld_shdw_mask(img: ee.ImageCollection) -> ee.ImageCollection:
    """Apply the cloud mask to each image in the collection.

    Args:
        img (ee.ImageCollection): image from add_cld_shdw_mask function

    Returns:
        ee.ImageCollection: collection with updated images
    """
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select("cloudmask").Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select("B.*").updateMask(not_cld_shdw)
