"""Utilities for MiniFrance dataset."""

RAW_CLASS_NAME_TO_LABEL = {
    "No information": 0,
    "Urban fabric": 1,
    "Industrial, commercial, public, military, private and transport units": 2,
    "Mine, dump and contruction sites": 3,
    "Artificial non-agricultural vegetated areas": 4,
    "Arable land (annual crops)": 5,
    "Permanent crops": 6,
    "Pastures": 7,
    "Complex and mixed cultivation patterns": 8,
    "Orchards at the fringe of urban classes": 9,
    "Forests": 10,
    "Herbaceous vegetation associations": 11,
    "Open spaces with little or no vegetation": 12,
    "Wetlands": 13,
    "Water": 14,
    "Clouds and shadows": 15,
}

RAW_LABELS_TO_RGB = {
    "No information": (0, 0, 0),
    "Urban fabric": (154, 163, 171),
    "Industrial, commercial, public, military, private and transport units": (207, 217, 227),
    "Mine, dump and contruction sites": (214, 211, 163),
    "Artificial non-agricultural vegetated areas": (157, 181, 130),
    "Arable land (annual crops)": (161, 161, 53),
    "Permanent crops": (201, 204, 20),
    "Pastures": (119, 247, 149),
    "Complex and mixed cultivation patterns": (222, 222, 106),
    "Orchards at the fringe of urban classes": (174, 194, 45),
    "Forests": (7, 138, 38),
    "Herbaceous vegetation associations": (39, 184, 72),
    "Open spaces with little or no vegetation": (105, 207, 128),
    "Wetlands": (109, 179, 232),
    "Water": (38, 151, 237),
    "Clouds and shadows": (0, 0, 0),
}

REMAP_LABELS = {
    "No information": ["Clouds and shadows", "No information"],
    "Urban": ["Urban fabric", "Industrial, commercial, public, military, private and transport units", "Mine, dump and contruction sites"],
    "Low vegetation": ["Pastures", "Open spaces with little or no vegetation"],
    "High vegetation": ["Orchards at the fringe of urban classes", "Forests", "Herbaceous vegetation associations", "Artificial non-agricultural vegetated areas"],
    "Crops": ["Arable land (annual crops)", "Permanent crops", "Complex and mixed cultivation patterns"],
    "Water": ["Wetlands", "Water"],
}

REMAP_LABELS_TO_RGB = {
    "No information": (0, 0, 0),
    "Urban": (154, 163, 171),
    "Low vegetation": (119, 247, 149),
    "High vegetation": (7, 138, 38),
    "Crops": (201, 204, 20),
    "Water": (38, 151, 237),
}

FEATURES_NAMES_TO_BAND_IDX = {
    "B1": 0,
    "B11": 1,
    "B12": 2,
    "B2": 3,
    "B3": 4,
    "B4": 5,
    "B5": 6,
    "B6": 7,
    "B7": 8,
    "B8": 9,
    "B8A": 10,
    "B9": 11,
    "VH": 12,
    "VV": 13,
}
