import numpy as np
import requests
from scipy.spatial import KDTree

from schemas.geometry import BoundingBox, PointMarker
from utils.map_service_utils import *

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

def get_map_data(bounding_box: BoundingBox, endpoint, alt_params={}):
    api_setup = map_config[endpoint]
    response = requests.get(
        api_setup["wms_root_url"],
        params={**api_setup["data"], **{"bbox": bounding_box.to_string_latlon(), "height":"1500", "width":"1500"}, **alt_params},
        stream=True
    )
    return response.content