import numpy as np
from shapely.geometry import Polygon, box
from typing import TypedDict

from schemas.geometry import BoundingBox

class IconConfig(TypedDict):
    bounds: Polygon
    resolution: float
    precision: int

REQ_THRESHOLD = 200
ICON_MODELS_CONFIG: dict[str, IconConfig] = {
    "central": {
        "bounds": box(0.0, 44.0, 17.5, 57.5),
        "resolution": 0.02,
        "precision": 3
    },
    "eu": {
        "bounds": box(-23.5, 29.5, 45, 70.5),
        "resolution": 0.0625,
        "precision": 3
    },
    "global": {
        "bounds": box(-180, -90, 180, 90),
        "resolution": 0.1,
        "precision": 2
    },
}

def generate_grid_points(bounding_box: BoundingBox) -> list[(float, float)]:
    config = None
    for _, v in ICON_MODELS_CONFIG.items():
        if bounding_box.geom.covered_by(v["bounds"]):
            print("Selected icon model:", _)
            config = v
            break
    lat_min, lon_min, lat_max, lon_max = bounding_box.bounds_yx()
    resolution = config['resolution']
    eps = 1e-6

    n_lat = np.ceil((lat_max - lat_min) / config['resolution'])
    n_lon = np.ceil((lon_max - lon_min) / config['resolution'])
    if n_lat * n_lon > REQ_THRESHOLD:
        resolution = np.round(np.sqrt((lat_max - lat_min) * (lon_max - lon_min) / REQ_THRESHOLD), config['precision'])
        print("Resolution adjusted to:", resolution)
    
    lats = np.round(np.arange(lat_min, lat_max+eps, resolution), config["precision"])
    lons = np.round(np.arange(lon_min, lon_max+eps, resolution), config["precision"])
    print("Generated points:", len(lats) * len(lons))
    
    lat_grid, lon_grid = np.meshgrid(lats, lons)
    lats = lat_grid.flatten()
    lons = lon_grid.flatten()

    return list(zip(lats, lons))

def describe_dominant_wind_direction(wind_directions: list[float]) -> str:
    threshold = 0.1
    compass_bins = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    bin_edges= np.arange(22.5, 360, 45)


    bin_indices = np.digitize(np.asarray(wind_directions) % 360, bin_edges) % len(compass_bins)
    bin_counts = np.bincount(bin_indices, minlength=len(compass_bins))
    bin_ratios = bin_counts / np.sum(bin_counts)

    significant_bins = np.where(bin_ratios >= threshold)[0]
    return ', '.join([f"{compass_bins[i]} in {bin_ratios[i]*100:.0f}%" for i in significant_bins])