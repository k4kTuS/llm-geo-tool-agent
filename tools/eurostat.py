from io import BytesIO

import numpy as np
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from PIL import Image
from pyproj import Transformer
from shapely.geometry import box
from typing import Optional, Type

from tools.base import GeospatialTool
from tools.input_schemas.base import BaseGeomInput
from schemas.geometry import BoundingBox
from utils.tool import get_map_data

EUROSTAT_GRID_SIZE = 1000

class EurostatPopulationTool(GeospatialTool):
    name: str = "get_eurostat_population_data"
    description: str = "Provides processed population data from eurostat for the bounding box."
    args_schema: Optional[Type[BaseModel]] = BaseGeomInput
    boundary = box(-20.0, 30.0, 40.0, 70.0)

    def _run(self, bounding_box: BoundingBox):
        # Snap the bounding box to the nearest grid cells and obtain number of grid cells for width and height
        bbox_snapped, dimensions = transform_snap_bbox(
            bounding_box,
            source_crs="EPSG:4326",
            target_crs="EPSG:3035",
            x_grid_size=EUROSTAT_GRID_SIZE,
            y_grid_size=EUROSTAT_GRID_SIZE
        )
        # Problem when retrieving only one grid cell - 1x1 pixel is not enough
        cell_duplication = 1
        if dimensions[0] == 1 and dimensions[1] == 1:
            dimensions = (2, 2)
            cell_duplication = 4
        
        # Download population layers
        total_data = get_map_data(
            bbox_snapped,
            "EUROSTAT_2021",
            {
                "layers": "total_population_eurostat_griddata_2021",
                "crs": "EPSG:3035",
                "width": dimensions[0],
                "height": dimensions[1]
            }
        )
        employed_data = get_map_data(
            bbox_snapped,
            "EUROSTAT_2021",
            {
                "layers": "employed_population_eurostat_griddata_2021",
                "crs": "EPSG:3035",
                "width": dimensions[0],
                "height": dimensions[1]
            }
        )
        # Calculate total and employed population
        arr_total = np.array(Image.open(BytesIO(total_data))) / cell_duplication
        arr_employed = np.array(Image.open(BytesIO(employed_data))) / cell_duplication
        total_sum = int(np.sum(arr_total))
        employed_sum = int(np.sum(arr_employed))
        return "## Eurostat data:\n"\
            + f"- Total population: {total_sum:,}\n"\
            + f"- Employed population: {employed_sum:,} ({(employed_sum / total_sum+1e-10):.2%})\n"\
    
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
