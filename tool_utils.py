import numpy as np
import requests

from io import BytesIO
from PIL import Image
from pyproj import Transformer
from scipy.spatial import KDTree
from shapely.geometry import Polygon

from config import *

# DEM
def count_elevation_zones(elevation_array):
    zone_counts = {name: 0 for _, _, name in elevation_ranges}  # Initialize counts

    for min_val, max_val, zone_name in elevation_ranges:
        # Count values in the current range
        count = np.sum((elevation_array >= min_val) & (elevation_array < max_val))
        zone_counts[zone_name] += count

    return zone_counts

# OLU
def get_color_counts(image, n_colors=None):
    pixels = np.array(image)
    pixels = pixels.reshape(-1, 3)

    pixels_view = pixels.view(np.dtype((np.void, pixels.dtype.itemsize * pixels.shape[1])))
    unique_pixels, counts = np.unique(pixels_view, return_counts=True)

    # Convert back to normal RGB tuples
    unique_pixels = unique_pixels.view(pixels.dtype).reshape(-1, 3)
    # Combine pixels and their counts
    pixel_counts = zip(map(tuple, unique_pixels), counts)
    # Prepare KDTree for color mapping
    LU_colors = list(LU_rgb_mapping.values())
    rgb_counts = {tuple(rgb): 0 for rgb in LU_colors}
    kdtree = KDTree(LU_colors)
    # Get color counts for LU colors
    for rgb, cnt in pixel_counts:
        if rgb in rgb_counts:
            rgb_counts[rgb] += cnt
        else:
            # Map missing colors using KDTree and add the count
            _, index = kdtree.query(rgb)
            closest_color = tuple(LU_colors[index])
            rgb_counts[closest_color] += cnt
    # Sort by counts
    sorted_pixel_counts = sorted([(k,v) for k,v in rgb_counts.items() if v != 0], reverse=True, key=lambda x: x[1])
    # Get the top n colors
    return sorted_pixel_counts[:n_colors]

# Other helpers
def get_map(coords, endpoint, alt_params={}):
    bbox = ','.join([str(v) for v in coords])
    
    api_setup = map_config[endpoint]
    response = requests.get(
        api_setup["wms_root_url"],
        params={**api_setup["data"], **{"bbox": bbox, "height":"1500", "width":"1500"}, **alt_params},
        stream=True
    )
    return Image.open(BytesIO(response.content))

def get_area(coords):
    # Initialize a Transformer to convert from WGS84 (EPSG:4326) to a projected system
    # EPSG:4326 is latitude/longitude in degrees, EPSG:3395 is Pseudo Mercator in meters
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    # Transform coordinates - first array containts longitudes, second latitudes
    lng, lat = transformer.transform([coords[1], coords[3]], [coords[0], coords[2]])
    # Create polygon ring
    ring = [(lng[0], lat[0]), (lng[0], lat[1]), (lng[1], lat[1]), (lng[1], lat[0])]
    polygon = Polygon(ring)
    # Polygon area in km2
    return polygon.area/1000000