from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from tools.input_schemas.base_schemas import BaseGeomInput
from utils.geometry_utils import BoundingBox
from utils.tool_utils import get_spoi_data


class SpoiTool(BaseTool):
    name: str = "get_smart_points_of_interest"
    description: str = "Get processed data about points of interest in the selected area."
    args_schema: Optional[Type[BaseModel]] = BaseGeomInput

    def _run(self, bounding_box: BoundingBox):
        spoi_data = get_spoi_data(bounding_box)
        return f"Number of points of interest: {len(spoi_data['features'])}"\
            + "\n\n" + "\n".join([f"{poi['properties']['cat'][str.rfind(poi['properties']['cat'], '#')+1:]} - {poi['properties']['label']}" for poi in spoi_data['features']])