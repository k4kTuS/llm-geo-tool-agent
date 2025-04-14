from typing import Optional, Type

import geopandas as gpd
import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from shapely.geometry import box, shape

from tools.base import GeospatialTool
from tools.input_schemas.poi import PoiInput
from schemas.geometry import BoundingBox
from utils.map_service import wfs_config


class PoiTool(GeospatialTool):
    name: str = "get_points_of_interest"
    description: str = (
        "Provides processed data about points of interest for the bounding box. "
        "It covers the western part of Czech Republic."
    )
    args_schema: Optional[Type[BaseModel]] = PoiInput
    boundary = box(12.4, 48.9, 14.5, 50.15)

    def _run(self, bounding_box: BoundingBox, categories: Optional[list[str]] = None):
        gpd_poi = get_poi_data(bounding_box)
        if gpd_poi.empty:
            return "No points of interest found in the area."
        category_counts = gpd_poi.groupby("category").agg(poi_count = ("label", "count")).sort_values(by="poi_count", ascending=False)
        return (
            f"Found the following point of interest categories:\n"
            + category_counts.to_markdown()
        )
    
def get_poi_data(bounding_box: BoundingBox) -> gpd.GeoDataFrame:
    response = requests.get(
        wfs_config["SPOI"]["wfs_root_url"],
        params={**wfs_config["SPOI"]["data"], **{"bbox": bounding_box.to_string_lonlat()}},
        stream=True
    )
    data = response.json()

    records = [
        {
            "label": ftr['properties']['label'],
            "category": ftr['properties']['cat'].split("#")[-1],
            "geometry": shape(ftr['geometry'])
        }
        for ftr in data['features']
    ]
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")