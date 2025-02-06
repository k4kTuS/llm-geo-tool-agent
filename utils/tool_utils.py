from io import BytesIO

import geopandas as gpd
import json
import numpy as np
import requests
import pandas as pd
from PIL import Image
from scipy.spatial import KDTree
import streamlit as st

from paths import DATA_DIR
from utils.geometry_utils import BoundingBox, PointMarker
from utils.map_service_utils import *

# DEM
def count_elevation_zones(elevation_array):
    zone_counts = {name: 0 for _, _, name in elevation_ranges}  # Initialize counts

    for min_val, max_val, zone_name in elevation_ranges:
        # Count values in the current range
        count = np.sum((elevation_array >= min_val) & (elevation_array < max_val))
        zone_counts[zone_name] += count

    return zone_counts

# OLU
def get_color_counts(image, rgb_mapping, n_colors=None):
    pixels = np.array(image)
    pixels = pixels.reshape(-1, 3)

    pixels_view = pixels.view(np.dtype((np.void, pixels.dtype.itemsize * pixels.shape[1])))
    unique_pixels, counts = np.unique(pixels_view, return_counts=True)

    # Convert back to normal RGB tuples
    unique_pixels = unique_pixels.view(pixels.dtype).reshape(-1, 3)
    # Combine pixels and their counts
    pixel_counts = zip(map(tuple, unique_pixels), counts)
    # Prepare KDTree for color mapping using only directly matched colors
    matched_colors = [clr for clr in map(tuple, unique_pixels) if clr in list(rgb_mapping.values())]

    rgb_counts = {tuple(rgb): 0 for rgb in matched_colors}
    kdtree = KDTree(matched_colors)
    # Get color counts for LU colors
    for rgb, cnt in pixel_counts:
        if rgb in rgb_counts:
            rgb_counts[rgb] += cnt
        else:
            # Map missing colors using KDTree and add the count
            _, index = kdtree.query(rgb)
            closest_color = tuple(matched_colors[index])
            rgb_counts[closest_color] += cnt
    # Sort by counts
    sorted_pixel_counts = sorted([(k,v) for k,v in rgb_counts.items() if v != 0], reverse=True, key=lambda x: x[1])
    # Get the top n colors
    return sorted_pixel_counts[:n_colors]

def find_square_for_marker(square_list, marker_point: PointMarker):
    m_lon = marker_point.x
    m_lat = marker_point.y
    for square in square_list:
        lat1, lon1, lat2, lon2 = map(float, square.split('_'))

        if lat1 <= m_lat <= lat2 and lon1 <= m_lon <= lon2:
            return square
    return None

# SPOI
def get_spoi_data(bounding_box: BoundingBox):
    # SPOI endpoint expects lon1, lat1, lon2, lat2
    response = requests.get(
        wfs_config["SPOI"]["wfs_root_url"],
        params={**wfs_config["SPOI"]["data"], **{"bbox": bounding_box.to_string_lonlat()}},
        stream=True
    )
    return response.json()

# Tourism
def get_region_tourism_data(bounding_box: BoundingBox):
    """
    Currently works only with Czech Republic region data.
    """
    gdf = gpd.GeoDataFrame.from_file(f'{DATA_DIR}/visitors.geojson')
    df = pd.read_csv(f'{DATA_DIR}/ciselnik_obci.csv', index_col='chodnota')
    regions = gdf[gdf.intersects(bounding_box.geom)]
    # No regions found
    if regions.empty:
        return None, None
    # Use only one region at first
    regions.loc[:, 'fid'] = regions.loc[:, 'fid'].astype(int)
    first_region = regions.iloc[0]
    props = json.loads(first_region.properties)
    name = df.loc[first_region.fid].text

    data = dict(sorted((k,v) for k,v in props.items() if is_number(v['all_guests'])))
    # No data for given region
    if len(data) == 0:
        return None, name

    return data, name

# Other helpers
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def get_map(bounding_box: BoundingBox, endpoint, alt_params={}):
    api_setup = map_config[endpoint]
    response = requests.get(
        api_setup["wms_root_url"],
        params={**api_setup["data"], **{"bbox": bounding_box.to_string_latlon(), "height":"1500", "width":"1500"}, **alt_params},
        stream=True
    )
    return Image.open(BytesIO(response.content))