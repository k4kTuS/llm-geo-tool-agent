from typing import Optional, Type

import geopandas as gpd
import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from shapely.geometry import box, shape

from tools.base import GeospatialTool
from tools.input_schemas.poi import PoiInput
from schemas.geometry import BoundingBox
from utils.encoder import get_encoder
from utils.map_service import wfs_config
from sentence_transformers.util import cos_sim

RELEVANCE_THRESHOLD = 0.5

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
        
        if not categories:
            return (
                f"Found the following point of interest categories:\n"
                + category_counts.to_markdown()
            )
        encoder = get_encoder()
        query_categories = [category.lower() for category in categories]
        poi_categories = [category.lower() for category in gpd_poi["category"].unique().tolist()]

        query_cat_embeddings = encoder.encode(query_categories, convert_to_tensor=True)
        poi_cat_embeddings = encoder.encode(poi_categories, convert_to_tensor=True)
        
        similarity = cos_sim(query_cat_embeddings, poi_cat_embeddings)
        relevant_poi_cats = []
        for i, poi_cat in enumerate(poi_categories):
            if max(similarity[:, i]) > RELEVANCE_THRESHOLD:
                relevant_poi_cats.append(poi_cat)
        
        return (
            "Found the following point of interest categories relevant for the query:\n"
            + category_counts[category_counts.index.isin(relevant_poi_cats)].to_markdown()
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