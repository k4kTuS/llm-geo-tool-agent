from io import BytesIO

import geopandas as gpd
import json
import numpy as np
import requests
import pandas as pd
from PIL import Image
from pyproj import Transformer
from scipy.spatial import KDTree
from shapely.geometry import Polygon, box
import streamlit as st

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

def find_square_for_marker(square_list, marker_coords):
    m_lon, m_lat = marker_coords
    for square in square_list:
        lat1, lon1, lat2, lon2 = map(float, square.split('_'))

        if lat1 <= m_lat <= lat2 and lon1 <= m_lon <= lon2:
            return square
    return None

# SPOI
def get_spoi_data(coords):
    # bbox is in lat1, lon1, lat2, lon2 order, SPOI endpoint expects lon1, lat1, lon2, lat2
    bbox = ','.join([str(v) for v in [coords[1], coords[0], coords[3], coords[2]]])
    response = requests.get(
        wfs_config["SPOI"]["wfs_root_url"],
        params={**wfs_config["SPOI"]["data"], **{"bbox": bbox}},
        stream=True
    )
    return response.json()

# Tourism
@st.cache_data
def get_region_tourism_data(coords):
    """
    Currently works only with Czech Republic region data.
    """
    gdf = gpd.GeoDataFrame.from_file('data/visitors.geojson')
    df = pd.read_csv('data/ciselnik_obci.csv', index_col='chodnota')
    bbox_lon_lat = [coords[1], coords[0], coords[3], coords[2]]
    regions = gdf[gdf.intersects(box(*bbox_lon_lat))]
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
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    # Transform coordinates - first array containts longitudes, second latitudes
    lng, lat = transformer.transform([coords[1], coords[3]], [coords[0], coords[2]])
    # Create polygon ring
    ring = [(lng[0], lat[0]), (lng[0], lat[1]), (lng[1], lat[1]), (lng[1], lat[0])]
    polygon = Polygon(ring)
    # Polygon area in km2
    return polygon.area/1000000

def get_area_gpd(coords):
    """
    Calculate the area of a bounding box in square kilometers using GeoPandas, estimating the correct UTM CRS.

    Args:
        coords: Bounding box coordinates in lat1, lon1, lat2, lon2 order.
    """
    gdf = gpd.GeoDataFrame({"geometry": [box(coords[1], coords[0], coords[3], coords[2])]}, crs="EPSG:4326")
    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)
    return gdf_utm.iloc[0].geometry.area/1000000
