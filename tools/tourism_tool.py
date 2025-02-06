from typing import Optional, Type

import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from tools.input_schemas.base_schemas import BaseGeomInput
from utils.geometry_utils import BoundingBox
from utils.tool_utils import get_region_tourism_data


class TourismTool(BaseTool):
    name: str = "get_tourism_data"
    description: str = "Get historical tourism data for regions based on given coordinates."
    args_schema: Optional[Type[BaseModel]] = BaseGeomInput

    def _run(self, bounding_box: BoundingBox):
        data, region_name = get_region_tourism_data(bounding_box)
    
        if data is None:
            if region_name is None:
                return "There is no existing tourism data for the selected region"
            return f"There is no existing tourism data for region {region_name}"

        pds = pd.DataFrame.from_dict(data, orient="index").loc[:, 'all_guests'].astype(float).astype(int)
        pds.index = pds.index.astype(int)

        tourism_data_string = f"Tourism data for region {region_name}:\n\n"\
            + "Number of all guests for recent years:\n"\
            + "\n".join([f"{k}: {v}" for k,v in pds.items()])
        return tourism_data_string
