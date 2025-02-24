import numpy as np
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from typing import Optional, Type

from tools.input_schemas.base_schemas import BaseGeomInput
from schemas.geometry import BoundingBox
from utils.tool_utils import get_map


class EurostatPopulationTool(BaseTool):
    name: str = "get_eurostat_population_data"
    description: str = "Get processed eurostat data about total population."
    args_schema: Optional[Type[BaseModel]] = BaseGeomInput

    def _run(self, bounding_box: BoundingBox):
        image = get_map(bounding_box, "EUROSTAT_2021", {"layer": "total_population_eurostat_griddata_2021"})
        total_population = int(np.sum(np.unique(np.array(image))))
        
        return f"Eurostat - Total population: {total_population}"
