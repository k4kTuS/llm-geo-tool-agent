import cv2
import numpy as np
import requests
from pyproj import Transformer
from shapely.geometry import box
from scipy.spatial import KDTree

from schemas.geometry import BoundingBox, PointMarker
from config.wms import wms_config

def detect_components(image_array, colors, connectivity=8, min_size=9):
    """
    Detects unique colors and connected components (zones) using OpenCV with a threshold on zone sizes.
    
    Args:
        image_array (np.ndarray): The input image as a NumPy array.
        colors (list): List of unique RGB tuples to count.
        connectivity (int): Connectivity for connected components (4 or 8).
        min_size (int): Minimum size of components in pixels to consider.
    
    Returns:
        dict: A dictionary mapping each RGB color to the number of connected components.
    """
    zones_per_color = {}

    for clr in colors:
        mask = cv2.inRange(image_array, np.array(clr), np.array(clr))
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity)
        component_sizes = stats[1:, cv2.CC_STAT_AREA]

        valid_count = np.sum(component_sizes >= min_size)
        zones_per_color[clr] = int(valid_count)
    return zones_per_color

def get_color_counts(image_array: np.ndarray):
    # Flatten to array of pixels
    pixels = image_array.reshape(-1, 3)
    # Detect unique colors using byte representation
    pixels_view = pixels.view(np.dtype((np.void, pixels.dtype.itemsize * pixels.shape[1])))
    unique_colors, counts = np.unique(pixels_view, return_counts=True)
    # Convert back to RGB tuples
    unique_colors = unique_colors.view(pixels.dtype).reshape(-1, 3)
    # Return as a dictionary
    return dict(zip(map(tuple, unique_colors), counts))

def get_mapped_color_counts(color_counts, matching_colors, n_colors=None):
    kdtree = KDTree(matching_colors)
    # Map unmatched colors and increase counts
    for rgb in set(color_counts.keys()) - set(matching_colors):
        _, index = kdtree.query(rgb)
        closest_color = tuple(matching_colors[index])
        color_counts[closest_color] += color_counts[rgb]
    # Sort by counts and filter to only matched colors
    sorted_pixel_counts =  sorted([(k,v) for k,v in color_counts.items() if k in matching_colors], reverse=True, key=lambda x: x[1])
    # Get the top n colors
    return sorted_pixel_counts[:n_colors]

def find_square_for_marker(square_list, point_marker: PointMarker):
    m_lon = point_marker.x
    m_lat = point_marker.y
    for square in square_list:
        lat1, lon1, lat2, lon2 = map(float, square.split('_'))

        if lat1 <= m_lat <= lat2 and lon1 <= m_lon <= lon2:
            return square
    return None

def get_map_data(bounding_box: BoundingBox, endpoint, alt_params={}):
    api_setup = wms_config[endpoint]
    response = requests.get(
        api_setup["wms_root_url"],
        params={**api_setup["data"], **{"bbox": bounding_box.to_string_latlon(), "height":"1500", "width":"1500"}, **alt_params},
        stream=True
    )
    return response.content

def transform_snap_bbox(bbox: BoundingBox, source_crs: str, target_crs: str, x_grid_size: int, y_grid_size: int):
    """
    Given a bounding box, source and target CRS, and grid cell sizes in the target CRS,
    this function snaps the bounds of the bounding box to the nearest grid cells and calculates
    the number of grid cells in each direction inside the bounding box.

    Args:
        bbox (BoundingBox): The bounding box to snap.
        source_crs (str): The source CRS of the bounding box.
        target_crs (str): The target CRS to which the bounding box should be transformed.
        x_grid_size (int): The size of the grid cells in the x-direction in the target CRS.
        y_grid_size (int): The size of the grid cells in the y-direction in the target CRS.
    Returns:
        tuple: A tuple containing the snapped bounding box and the number of grid cells in each direction.
    """
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    bounds_target = transformer.transform_bounds(*bbox.bounds_lonlat())
    bounds_target_snapped = (
        np.floor(bounds_target[0] / x_grid_size) * x_grid_size,
        np.floor(bounds_target[1] / y_grid_size) * y_grid_size,
        np.ceil(bounds_target[2] / x_grid_size) * x_grid_size,
        np.ceil(bounds_target[3] / y_grid_size) * y_grid_size
    )
    bbox_snapped = BoundingBox(wkt=box(*bounds_target_snapped).wkt)
    x_grid_cells = int((bounds_target_snapped[2] - bounds_target_snapped[0]) / x_grid_size)
    y_grid_cells = int((bounds_target_snapped[3] - bounds_target_snapped[1]) / y_grid_size)

    return bbox_snapped, (x_grid_cells, y_grid_cells)
