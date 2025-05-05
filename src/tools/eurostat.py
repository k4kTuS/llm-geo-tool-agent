from io import BytesIO

import numpy as np
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from PIL import Image
from shapely.geometry import box
from typing import Optional, Type

from tools.base import GeospatialTool
from tools.input_schemas.base import BaseGeomInput
from schemas.data import DataResponse
from schemas.geometry import BoundingBox
from utils.map_analysis import get_map_data, transform_snap_bbox

EUROSTAT_GRID_SIZE = 1000

class EurostatPopulationTool(GeospatialTool):
    name: str = "get_eurostat_population_data"
    description: str = "Provides total population and employed population data from eurostat for the bounding box."
    args_schema: Optional[Type[BaseModel]] = BaseGeomInput
    response_format: str = "content_and_artifact"
    boundary = box(-20.0, 30.0, 40.0, 70.0)

    def _run(self, bounding_box: BoundingBox):
        # Snap the bounding box to the nearest grid cells and obtain number of grid cells for width and height
        bbox_snapped, dimensions = transform_snap_bbox(
            bounding_box,
            target_crs="EPSG:3035",
            x_grid_size=EUROSTAT_GRID_SIZE,
            y_grid_size=EUROSTAT_GRID_SIZE
        )
        # Problem when retrieving only one grid cell in any direction
        cell_duplication = 1
        if dimensions[0] == 1 or dimensions[1] == 1:
            dimensions = tuple(d * 2 for d in dimensions)
            cell_duplication = 4
        
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
            bbox_snapped,
            "EUROSTAT_2021",
            {
                "layers": "employed_population_eurostat_griddata_2021",
                "width": dimensions[0],
                "height": dimensions[1]
            }
        )
        # Calculate total and employed population
        arr_total = np.array(Image.open(BytesIO(total_data))) / cell_duplication
        arr_employed = np.array(Image.open(BytesIO(employed_data))) / cell_duplication
        total_sum = int(np.sum(arr_total))
        employed_sum = int(np.sum(arr_employed))

        output = "## Eurostat data:\n"\
            + f"- Total population: {total_sum:,}\n"\
            + f"- Employed population: {employed_sum:,} ({(employed_sum / total_sum+1e-10):.2%})"
        data_output = DataResponse(
            name="Eurostat Population Data",
            source="Eurostat Census 2021",
            data_type="text",
            data=output,
            show_data=True,
        )
        return output, data_output
