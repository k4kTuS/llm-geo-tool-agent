import cv2
import numpy as np
import requests
from scipy.spatial import KDTree

from schemas.geometry import BoundingBox, PointMarker
from utils.map_service import *

def get_color_counts(image, rgb_mapping, n_colors=None):
    pixels = np.array(image).reshape(-1, 3)

    # Count uniques using byte representation
    pixels_view = pixels.view(np.dtype((np.void, pixels.dtype.itemsize * pixels.shape[1])))
    unique_pixels, counts = np.unique(pixels_view, return_counts=True)
    # Convert back to RGB tuples
    unique_pixels = unique_pixels.view(pixels.dtype).reshape(-1, 3)
    # Combine pixels and their counts
    pixel_counts = dict(zip(map(tuple, unique_pixels), counts))
    # Prepare KDTree for color mapping using only directly matched colors
    matched_colors = [clr for clr in pixel_counts.keys() if clr in list(rgb_mapping.values())]
    kdtree = KDTree(matched_colors)
    # Map unmatched colors and increase counts
    for rgb in set(pixel_counts.keys()) - set(matched_colors):
        _, index = kdtree.query(rgb)
        closest_color = tuple(matched_colors[index])
        pixel_counts[closest_color] += pixel_counts[rgb]
    # Sort by counts and filter to only matched colors
    sorted_pixel_counts =  sorted([(k,v) for k,v in pixel_counts.items() if k in matched_colors], reverse=True, key=lambda x: x[1])
    # Get the top n colors
    return sorted_pixel_counts[:n_colors]

def count_components(image_array, colors, connectivity=8):
    """
    Counts connected components (zones) of each RGB color using OpenCV.
    
    Args:
        image_array (np.ndarray): The input image as a NumPy array.
        unique_colors (list): List of unique RGB tuples to count.
        connectivity (int): Connectivity for connected components (4 or 8).
    
    Returns:
        dict: A dictionary mapping each RGB color to the number of connected components.
    """
    zones_per_color = {}
    for clr in colors:
        mask = cv2.inRange(image_array, np.array(clr), np.array(clr))
        num_labels, _ = cv2.connectedComponents(mask, connectivity=connectivity)
        # Subtract 1 to remove background from count
        zones_per_color[tuple(clr)] = num_labels - 1
    return zones_per_color

def find_square_for_marker(square_list, point_marker: PointMarker):
    m_lon = point_marker.x
    m_lat = point_marker.y
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