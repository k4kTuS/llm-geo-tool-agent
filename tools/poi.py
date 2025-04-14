from typing import Optional, Type

import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from shapely.geometry import box

from tools.base import GeospatialTool
from tools.input_schemas.base import BaseGeomInput
from schemas.geometry import BoundingBox
from utils.map_service import wfs_config


class SpoiTool(GeospatialTool):
    name: str = "get_smart_points_of_interest"
    description: str = "Provides processed data about points of interest for the bounding box."
    args_schema: Optional[Type[BaseModel]] = BaseGeomInput
    boundary = box(12.4, 48.9, 14.5, 50.15)

    def _run(self, bounding_box: BoundingBox):
        spoi_data = get_spoi_data(bounding_box)
        return f"Number of points of interest: {len(spoi_data['features'])}"\
            + "\n\n" + "\n".join([f"{poi['properties']['cat'][str.rfind(poi['properties']['cat'], '#')+1:]} - {poi['properties']['label']}" for poi in spoi_data['features']])
    
def get_spoi_data(bounding_box: BoundingBox):
    # SPOI endpoint expects lon1, lat1, lon2, lat2
    response = requests.get(
        wfs_config["SPOI"]["wfs_root_url"],
        params={**wfs_config["SPOI"]["data"], **{"bbox": bounding_box.to_string_lonlat()}},
        stream=True
    )
    return response.json()