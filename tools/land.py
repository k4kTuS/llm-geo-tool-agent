from io import BytesIO
from typing import Optional, Type

import numpy as np
from langchain_core.tools import BaseTool
import pandas as pd
from PIL import Image
from pydantic import BaseModel
from rasterio.io import MemoryFile
from shapely import box
from typing import Optional, Type

from tools.base import GeospatialTool
from tools.input_schemas.base import BaseGeomInput
from schemas.geometry import BoundingBox
from schemas.data import DataResponse
from utils.tool import get_map_data, get_color_counts, get_mapped_color_counts, detect_components
from utils.map_service import LC_rgb_mapping, LU_rgb_mapping, rgb_LC_mapping, rgb_LU_mapping, elevation_ranges


class LandCoverTool(GeospatialTool):
    name: str = "land_cover_tool"
    description: str = (
        "Provides processed land cover data for the bounding box. "
        "Zone types are based on physical surface characteristics."
        "The result is a summary of land cover types, area sizes, and counts."
    )
    args_schema: Optional[Type[BaseModel]] = BaseGeomInput
    response_format: str = "content_and_artifact"
    boundary = box(-20.0, 30.0, 40.0, 70.0)

    def _run(self, bounding_box: BoundingBox):
        map_data = get_map_data(bounding_box, "OLU_EU", {"layers": "olu_obj_lc"})
        image = Image.open(BytesIO(map_data))
        image_data = np.array(image)
        n_pixels = len(image.getdata())

        color_counts = get_color_counts(image_data)
        matching_colors = [k for k in color_counts.keys() if k in list(LC_rgb_mapping.values())]
        component_counts = detect_components(image_data, matching_colors, connectivity=8, min_size=9)
        filtered_colors = [k for k,v in component_counts.items() if v > 0]
        rgb_counts = get_mapped_color_counts(color_counts, filtered_colors)
        
        bbox_area = bounding_box.area
        unit = "km²"
        if bbox_area < 1:
            bbox_area *= 1_000_000
            unit = "m²"

        zone_data = []
        for rgb, count in rgb_counts:
            zone_name = rgb_LC_mapping[rgb]
            ratio = count / n_pixels
            area = ratio * bbox_area

            zone_data.append({
                "Zone Name": zone_name,
                f"Area ({unit})": round(area, 4),
                "Area Ratio (%)": round(ratio * 100, 4),
                "Zone Count": component_counts[rgb],
                "small": ratio < 0.01
            })
        df = pd.DataFrame(zone_data)
        df_main = df[~df['small']].drop(columns=['small'])
        df_small = df[df['small']].drop(columns=['small'])
        
        output = f"### Map Area: {bbox_area:.2f} {unit}\n\n"
        if not df_main.empty:
            output += "#### Land Cover Summary\n\n"
            output += df_main.to_markdown(index=False)
            output += "\n\n"

        if not df_small.empty:
            output += "#### Small Zones (<1%)\n\n"
            output += df_small.to_markdown(index=False)

        dataset = DataResponse(
            name="Land Cover Data",
            source="Open Land Use Map",
            data_type="dataframe",
            data=df,
            show_data=True,
        )
        return output, dataset


class LandUseTool(GeospatialTool):
    name: str = "land_use_tool"
    description: str = (
        "Provides processed land use data for the bounding box. "
        "Zone types are based on human activities and planning. "
        "The result is a summary of land use types, area sizes, and counts."
    )
    args_schema: Optional[Type[BaseModel]] = BaseGeomInput
    response_format: str = "content_and_artifact"
    boundary = box(-20.0, 30.0, 40.0, 70.0)

    def _run(self, bounding_box: BoundingBox):
        map_data = get_map_data(bounding_box, "OLU_EU")
        image = Image.open(BytesIO(map_data))
        image_data = np.array(image)
        n_pixels = len(image.getdata())

        color_counts = get_color_counts(image_data)
        matching_colors = [k for k in color_counts.keys() if k in list(LU_rgb_mapping.values())]
        component_counts = detect_components(image_data, matching_colors, connectivity=8, min_size=9)
        filtered_colors = [k for k,v in component_counts.items() if v > 0]
        rgb_counts = get_mapped_color_counts(color_counts, filtered_colors)

        bbox_area = bounding_box.area
        unit = "km²"
        if bbox_area < 1:
            bbox_area *= 1_000_000
            unit = "m²"

        zone_data = []
        for rgb, count in rgb_counts:
            zone_name = rgb_LU_mapping[rgb]
            ratio = count / n_pixels
            area = ratio * bbox_area

            zone_data.append({
                "Zone Name": zone_name,
                f"Area ({unit})": round(area, 4),
                "Area Ratio (%)": round(ratio * 100, 4),
                "Zone Count": component_counts[rgb],
                "small": ratio < 0.01
            })
        df = pd.DataFrame(zone_data)
        df_main = df[~df['small']].drop(columns=['small'])
        df_small = df[df['small']].drop(columns=['small'])
        
        output = f"### Map Area: {bbox_area:.2f} {unit}\n\n"
        if not df_main.empty:
            output += "#### Land Use Summary\n\n"
            output += df_main.to_markdown(index=False)
            output += "\n\n"

        if not df_small.empty:
            output += "#### Small Zones (<1%)\n\n"
            output += df_small.to_markdown(index=False)
        
        dataset = DataResponse(
            name="Land Use Data",
            source="Open Land Use Map",
            data_type="dataframe",
            data=df,
            show_data=True,
        )
        return output, dataset
    

class ElevationTool(GeospatialTool):
    name: str = "elevation_tool"
    description: str = "Provides processed data from digital elevation model, including basic statistics and division into elevation zones."
    args_schema: Optional[Type[BaseModel]] = BaseGeomInput
    boundary = box(11.86, 48.52, 19.02, 51.11)

    def _run(self, bounding_box: BoundingBox):
        map_data = get_map_data(bounding_box, "DEM_MASL")
        
        with MemoryFile(map_data) as memfile:
            with memfile.open() as dataset:
                raster = dataset.read(1)
                # Mask nodata values
                nodata_value = dataset.nodata
                if nodata_value is not None:
                    raster = np.ma.masked_equal(raster, nodata_value)
                stats = {
                    "min": raster.min(),
                    "max": raster.max(),
                    "mean": raster.mean(),
                    "std": raster.std(),
                }

        zones_data = count_elevation_zones(raster)
        n_pixels = len(raster.flatten())
        bbox_area = bounding_box.area
        zones_ratios = {k: v / n_pixels for k, v in zones_data.items()}
        
        return (
            f"### Elevation Statisticts:\n"
            f"Average elevation: {stats["mean"]:.2f} meters\n"
            f"Max elevation: {stats["max"]:.2f} meters\n"
            f"Min elevation: {stats["mean"]:.2f} meters\n"
            f"Standard deviation: {stats["std"]:.2f} meters\n"
            "### Elevation Zones Coverage:\n"
            + "\n".join([f"{k}: {v * bbox_area:.2f} km2 ({v*100:.2f}%)" for k, v in zones_ratios.items() if v != 0])
        )

def count_elevation_zones(elevation_array):
    zone_counts = {}
    # Count pixels in each elevation range
    for min_val, max_val, zone_name in elevation_ranges:
        zone_label = f"{zone_name} ({min_val}-{max_val} m)"
        count = np.sum((elevation_array >= min_val) & (elevation_array < max_val))
        zone_counts[zone_label] = count

    return zone_counts