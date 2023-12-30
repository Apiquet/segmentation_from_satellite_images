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
