from io import BytesIO

import numpy as np
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from PIL import Image
from pyproj import Transformer
from pyproj.enums import TransformDirection
from shapely.geometry import box
from typing import Optional, Type

from tools.input_schemas.base_schemas import BaseGeomInput
from schemas.geometry import BoundingBox
from utils.tool_utils import get_map_data

EUROSTAT_GRID_SIZE = 1000

class EurostatPopulationTool(BaseTool):
    name: str = "get_eurostat_population_data"
    description: str = "Provides processed population data from eurostat for the bounding box."
    args_schema: Optional[Type[BaseModel]] = BaseGeomInput

    def _run(self, bounding_box: BoundingBox):
        # Snap the bounding box to the nearest grid cells and obtain number of grid cells for width and height
        bbox_snapped, dimensions = transform_snap_bbox(
            bounding_box,
            source_crs="EPSG:4326",
            target_crs="EPSG:3035",
            x_grid_size=EUROSTAT_GRID_SIZE,
            y_grid_size=EUROSTAT_GRID_SIZE
        )
        # Download population layers
        total_data = get_map_data(
            bbox_snapped,
            "EUROSTAT_2021",
            {
                "layers": "total_population_eurostat_griddata_2021",
                "width": dimensions[0],
                "height": dimensions[1]
            }
        )
        employed_data = get_map_data(
            bounding_box,
            "EUROSTAT_2021",
            {
                "layers": "employed_population_eurostat_griddata_2021",
                "width": dimensions[0],
                "height": dimensions[1]
            }
        )
        # Calculate total and employed population
        arr_total = np.array(Image.open(BytesIO(total_data)))
        arr_employed = np.array(Image.open(BytesIO(employed_data)))
        return "## Eurostat data:\n"\
            + f"- Total population: {int(np.sum(arr_total))}\n"\
            + f"- Employed population: {int(np.sum(arr_employed))}"
    
def transform_snap_bbox(bbox: BoundingBox, source_crs: str, target_crs: str, x_grid_size: int, y_grid_size: int):
    """
    Given a bounding box, source and target CRS, and grid cell sizes in the target CRS,,
    this function snaps the bounds of the bounding box to the nearest grid cells and calculates
    the number of grid cells in each direction inside the bounding box.
    """
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    bounds_target = transformer.transform_bounds(*bbox.bounds_lonlat(), direction=TransformDirection.FORWARD)
    bounds_target_snapped = (
        np.floor(bounds_target[0] / x_grid_size) * x_grid_size,
        np.floor(bounds_target[1] / y_grid_size) * y_grid_size,
        np.ceil(bounds_target[2] / x_grid_size) * x_grid_size,
        np.ceil(bounds_target[3] / y_grid_size) * y_grid_size
    )
    bounds_source_snapped = transformer.transform_bounds(*bounds_target_snapped, direction=TransformDirection.INVERSE)
    bbox_snapped = BoundingBox(wkt=box(*bounds_source_snapped).wkt)

    x_grid_cells = int((bounds_target_snapped[2] - bounds_target_snapped[0]) / x_grid_size)
    y_grid_cells = int((bounds_target_snapped[3] - bounds_target_snapped[1]) / y_grid_size)

    return bbox_snapped, (x_grid_cells, y_grid_cells)
