from io import BytesIO

import numpy as np
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from PIL import Image
from typing import Optional, Type

from tools.input_schemas.base_schemas import BaseGeomInput
from schemas.geometry import BoundingBox
from utils.tool_utils import get_map_data


class EurostatPopulationTool(BaseTool):
    name: str = "get_eurostat_population_data"
    description: str = "Provides processed population data from eurostat for the bounding box."
    args_schema: Optional[Type[BaseModel]] = BaseGeomInput

    def _run(self, bounding_box: BoundingBox):
        map_data = get_map_data(bounding_box, "EUROSTAT_2021", {"layer": "total_population_eurostat_griddata_2021"})
        image = Image.open(BytesIO(map_data))
        total_population = int(np.sum(np.unique(np.array(image))))
        
        return f"Eurostat - Total population: {total_population}"