import json
from typing import Optional, Type

import numpy as np
import pandas as pd
import pickle
import geopandas as gpd
import osmnx as ox
import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from shapely.geometry import box, shape

from tools.base import GeospatialTool
from tools.input_schemas.poi import PoiInput, OSMInput
from schemas.data import DataResponse
from schemas.geometry import BoundingBox
from utils.encoder import get_encoder
from config.wfs import wfs_config
from sentence_transformers.util import cos_sim
from config.project_paths import DATA_DIR

class OSMTool(GeospatialTool):
    name: str = "get_open_street_map_features"
    description: str = (
        "Retrieves points of interest (POIs) from OpenStreetMap within a bounding box based on a natural language query. "
        "The input query must be a descriptive sentence, not a tag or list of tags. "
        "The tool uses semantic similarity to match the query to relevant OSM tag combinations, which are then used to fetch features. "
        "Returns a summary of matched features grouped by OSM tag categories."
    )
    args_schema: Optional[Type[BaseModel]] = OSMInput

    def _run(self, bounding_box: BoundingBox, query: str, topk: int = 20):
        # Load prepared descriptions and pre-computed embeddings with metadata
        with open(DATA_DIR / "osm_tag_descriptions.json", "r") as f:
            tag_descriptions = json.load(f)
        with open(DATA_DIR / "osm_tag_embeddings.pkl", "rb") as f:
            tag_embeddings = pickle.load(f)
        # Compute similarity scores and select topk relevant tags
        encoder = get_encoder()
        query_embedding = encoder.encode(
            sentences=query,
            prompt="query: ",
            convert_to_numpy=True
        )
        similarities = cos_sim(query_embedding, np.array([tag['embedding'] for tag in tag_embeddings]))[0]
        top_indices = similarities.cpu().numpy().argsort()[::-1][:topk]
        top_tags = [tag_embeddings[i] for i in top_indices]
        # Build tags dict compatible with osmnx querying
        tags = {}
        for v in top_tags:
            if v['key'] not in tags:
                tags[v['key']] = []
            tags[v['key']].append(v['value'])
        # Retrieve features based on tags
        gdf = ox.features.features_from_bbox(bounding_box.bounds(), tags=tags)
        if gdf.empty:
            return "Found no features from OpenStreetMap relevant to the query."
        # Build textual summary
        summary = f"Retrieved {len(gdf)} unique features. Below are their characteristics (the categories overlap):"

        for key in tags:
            if key in gdf.columns:
                value_counts = gdf[key].value_counts().reset_index()
                value_counts['description'] = value_counts[key].apply(lambda x: tag_descriptions.get(f"{key}={x}", "No description available."))
                summary = summary + f"\n### {key} tag data:\n"
                summary = summary + value_counts.to_markdown(index=False)
        return summary

class PoiTool(GeospatialTool):
    name: str = "get_points_of_interest"
    description: str = (
        "Retrieves processed data about points of interest for the bounding box. "
        "Provides only a subset of features from OpenStreetMap. "
        "Accepts an optional list of categories and a relevance threshold for fuzzy matching. "
        "Has the geospatial coverage for the western part of Czech Republic. "
        "Is faster than general OpenStreetMap querying."
    )
    args_schema: Optional[Type[BaseModel]] = PoiInput
    response_format: str = "content_and_artifact"
    boundary = box(12.4, 48.9, 14.5, 50.15)

    def _run(self, bounding_box: BoundingBox, categories: Optional[list[str]] = None, relevance_threshold: Optional[float] = None):
        if relevance_threshold is None:
            relevance_threshold = 0.8

        gpd_poi = get_poi_data(bounding_box)
        if gpd_poi.empty:
            return "No points of interest from the available subset exist in the area.", None
        category_counts = gpd_poi.groupby("category").agg(count = ("label", "count")).sort_values(by="count", ascending=False)
        
        if not categories:
            output =  (
                f"### Points of interest:\n"
                + category_counts.to_markdown()
            )
            data_output = DataResponse(
                name="Points of Interest",
                source="OpenStreetMap",
                data_type="dataframe",
                data=category_counts.reset_index(),
                show_data=True,
            )
            return output, data_output
        encoder = get_encoder()
        query_categories = [category.lower() for category in categories]
        poi_categories = [category.lower() for category in gpd_poi["category"].unique().tolist()]

        query_cat_embeddings = encoder.encode(query_categories, convert_to_tensor=True)
        poi_cat_embeddings = encoder.encode(poi_categories, convert_to_tensor=True)
        
        similarity = cos_sim(query_cat_embeddings, poi_cat_embeddings)
        relevant_poi_cats = []
        for i, poi_cat in enumerate(poi_categories):
            if max(similarity[:, i]) > relevance_threshold:
                relevant_poi_cats.append(poi_cat)
        
        res_gdf = category_counts[category_counts.index.isin(relevant_poi_cats)]
        if res_gdf.empty:
            return f"Retrieved {len(gpd_poi)} points of interest, but none are above the {relevance_threshold} relevance threshold you defined. You can consider retrying with lower threshold.", None
        output = (
            "### Filtered points of interest:\n"
            + res_gdf.to_markdown()
        )
        data_output = DataResponse(
            name="Points of Interest",
            source="OpenStreetMap",
            data_type="dataframe",
            data=res_gdf.reset_index(),
            show_data=True,
        )
        return output, data_output
    
def get_poi_data(bounding_box: BoundingBox) -> gpd.GeoDataFrame:
    response = requests.get(
        wfs_config["SPOI"]["wfs_root_url"],
        params={**wfs_config["SPOI"]["data"], **{"bbox": bounding_box.to_string()}},
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
    if not records:
        return gpd.GeoDataFrame(columns=["label", "category", "geometry"], crs="EPSG:4326")
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")