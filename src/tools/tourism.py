from typing import Optional, Type

import geopandas as gpd
import json
import numpy as np
import pandas as pd
import pyproj
from shapely.ops import transform
from shapely import box
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from config.project_paths import DATA_DIR
from tools.base import GeospatialTool
from tools.input_schemas.base import BaseGeomInput
from schemas.geometry import BoundingBox
from schemas.data import DataResponse

TOP_K = 10
YEAR_COLUMNS = [str(year) for year in range(2012, 2024)]
NO_DATA_STRINGS = ['-', '.', 'x', 'i.d.']

class TourismTool(GeospatialTool):
    name: str = "get_tourism_data"
    description: str = (
        "Retrieves historical tourism data for municipalities intersecting a given geographic bounding box. "
        "Available only for regions within the Czech Republic. "
        "Returns a yearly summary of total tourist visits from 2012 to 2023 for the intersecting municipalities."
    )
    args_schema: Optional[Type[BaseModel]] = BaseGeomInput
    response_format: str = "content_and_artifact"
    boundary = box(12.09,48.55,18.87,51.06)

    def _run(self, bounding_box: BoundingBox):
        municipality_data = get_region_tourism_data(bounding_box)    
        if municipality_data is None:
            return "There is no existing tourism data for the selected area", None
        n_municipalities = len(municipality_data)
        # Drop regions with no data
        municipality_data = municipality_data.dropna(subset=YEAR_COLUMNS, how='all')
        if municipality_data.empty:
            return "There is no existing tourism data for the selected area", None
        # Create data sumaries
        agg_data = municipality_data[YEAR_COLUMNS].multiply(municipality_data['municipality_ratio'], axis=0).sum(axis=0).map(lambda x: int(x))
        municipality_data['sort_key'] = municipality_data[YEAR_COLUMNS].mean(axis=1).multiply(municipality_data['municipality_ratio'], axis=0)
        top_k = municipality_data.nlargest(TOP_K, 'sort_key').filter(items=['municipality', 'aoi_ratio', 'municipality_ratio']+YEAR_COLUMNS)

        data_response = DataResponse(
            name=f"Tourism data for {TOP_K} most relevant municipalities",
            source="Czech Statistical Office",
            data_type="dataframe",
            data=top_k,
            show_data=True
        )

        return "## Tourism data for the selected area:\n"\
            + f"Total intersected municipalities: {n_municipalities}\n"\
            + f"Municipalities with no data: {n_municipalities - len(municipality_data)}\n\n"\
            + "### 1. Aggregated data (Estimated values):\n"\
            + "These are **estimated values** calculated using data from all municipalities that intersect the area of interest (AOI).\n"\
            + "- For each municipality that overlaps with the AOI, its **yearly guest numbers** are scaled by how much of the municipality lies inside the AOI.\n\n"\
            + agg_data.to_markdown(headers=['Year', 'Weighted sum of guests'])\
            + f"\n\n### 2. Detailed information:\n"\
            + f"This section provides raw data for at most {TOP_K} intersecting municipalities, based on their aoi_ratio and all guest values. Intersecting municipalities with no data are excluded."\
            + "**Columns:**\n"\
            + "- **Yearly guest data**: Total annual guest numbers for the entire municipality.\n"\
            + "- **aoi_ratio**: The **percentage of the AOI** that lies within the municipality.\n"\
            + "- **municipality_ratio**: The **percentage of the municipality** that lies within the AOI.\n"\
            + "- **NaN** values indicate missing data.\n\n"\
            + top_k.to_markdown(index=False), data_response

def get_region_tourism_data(bounding_box: BoundingBox):
    # Load data
    gdf = gpd.GeoDataFrame.from_file(f'{DATA_DIR}/visitors.geojson')
    df = pd.read_csv(f'{DATA_DIR}/ciselnik_obci.csv', index_col='chodnota')
    # Calculate intersections and transform from coordinates
    gdf['intersection_geom'] = gdf.intersection(bounding_box.geom)
    utm_crs = gdf.estimate_utm_crs()
    transformer = pyproj.Transformer.from_crs(gdf.crs, utm_crs, always_xy=True)
    bbox_area = transform(transformer.transform, bounding_box.geom).area
    gdf['intersection_geom'] = gdf['intersection_geom'].apply(lambda geom: transform(transformer.transform, geom))
    gdf = gdf.to_crs(utm_crs)
    # Filter out non-intersecting regions
    gdf = gdf[~gdf.intersection_geom.is_empty]
    if gdf.empty:
        return None
    gdf['aoi_ratio'] = gdf['intersection_geom'].area / bbox_area
    gdf['municipality_ratio'] = gdf['intersection_geom'].area / gdf['geometry'].area
    # Parse all guests data from the properties
    properties_df = gdf['properties'].apply(lambda x: dict(sorted(json.loads(x).items()))).apply(pd.Series)
    # Build response dataframe
    tourism_df = properties_df.map(lambda x: float(x['all_guests']) if x['all_guests'] not in NO_DATA_STRINGS else np.nan)
    tourism_df['municipality'] = gdf['fid'].astype(int).map(df['text'])
    tourism_df['aoi_ratio'] = gdf['aoi_ratio']
    tourism_df['municipality_ratio'] = gdf['municipality_ratio']
    return tourism_df