from io import BytesIO
from typing import Optional, Type

import numpy as np
from langchain_core.tools import BaseTool
from PIL import Image
from pydantic import BaseModel
from typing import Optional, Type

from tools.input_schemas.base_schemas import BaseGeomInput
from schemas.geometry import BoundingBox
from utils.tool_utils import get_map_data, get_color_counts
from utils.map_service_utils import LC_rgb_mapping, LU_rgb_mapping, rgb_LC_mapping, rgb_LU_mapping, elevation_ranges


class LandCoverTool(BaseTool):
    name: str = "land_cover_tool"
    description: str = (
        "Provides processed land cover data for the bounding box. "
        "Zone types are based on physical surface characteristics."
        "The result is a summary of land cover types and their respective areas."
    )
    args_schema: Optional[Type[BaseModel]] = BaseGeomInput

    def _run(self, bounding_box: BoundingBox):
        map_data = get_map_data(bounding_box, "OLU_EU", {"layers": "olu_obj_lc"})
        image = Image.open(BytesIO(map_data))
        
        n_pixels = len(image.getdata())
        rgb_counts = get_color_counts(image, LC_rgb_mapping)
        
        land_uses = [rgb_LC_mapping[rgb] for rgb,_ in rgb_counts]
        land_ratios = [cnt/n_pixels for _,cnt in rgb_counts]

        bbox_area = bounding_box.area
        unit = "km squared"
        if bbox_area < 1:
            bbox_area *= 1000_000
            unit = "m squared"
        
        zones_data = []
        small_zones_data = []
        for lu, ratio in zip(land_uses, land_ratios):
            if ratio < 0.01:
                small_zones_data.append(f"{lu} - Area: {ratio*bbox_area:.4f} {unit} ({ratio*100:.4f}%)")
            else:
                zones_data.append(f"{lu} - Area: {ratio*bbox_area:.2f} {unit} ({ratio*100:.2f}%)")

        return f"Map Area: {bbox_area:.2f} {unit}\n\n"\
            + "Land cover information:\n"\
            + "\n".join(zones_data)\
            + "\n\n" + "Land cover information for small zones:\n"\
            + "\n".join(small_zones_data)


class LandUseTool(BaseTool):
    name: str = "land_use_tool"
    description: str = (
        "Provides processed land use data for the bounding box. "
        "Zone types are based on human activities and planning. "
        "The result is a summary of land use types and their respective areas."
    )
    args_schema: Optional[Type[BaseModel]] = BaseGeomInput

    def _run(self, bounding_box: BoundingBox):
        map_data = get_map_data(bounding_box, "OLU_EU")
        image = Image.open(BytesIO(map_data))
        
        n_pixels = len(image.getdata())
        rgb_counts = get_color_counts(image, LU_rgb_mapping)
        
        land_uses = [rgb_LU_mapping[rgb] for rgb,_ in rgb_counts]
        land_ratios = [cnt/n_pixels for _,cnt in rgb_counts]

        bbox_area = bounding_box.area
        unit = "km squared"
        if bbox_area < 1:
            bbox_area *= 1000_000
            unit = "m squared"
        
        zones_data = []
        small_zones_data = []
        for lu, ratio in zip(land_uses, land_ratios):
            if ratio < 0.01:
                small_zones_data.append(f"{lu} - Area: {ratio*bbox_area:.4f} {unit} ({ratio*100:.4f}%)")
            else:
                zones_data.append(f"{lu} - Area: {ratio*bbox_area:.2f} {unit} ({ratio*100:.2f}%)")

        return f"Map Area: {bbox_area:.2f} {unit}\n\n"\
            + "Land use information:\n"\
            + "\n".join(zones_data)\
            + "\n\n" + "Land use information for small zones:\n"\
            + "\n".join(small_zones_data)
    

class ElevationTool(BaseTool):
    name: str = "elevation_tool"
    description: str = "Get processed data from digital elevation model."
    args_schema: Optional[Type[BaseModel]] = BaseGeomInput

    def _run(self, bounding_box: BoundingBox):
        map_data = get_map_data(bounding_box, "DEM_MASL")
        image = Image.open(BytesIO(map_data))
        elevations = np.array(image)
        zones_data = count_elevation_zones(elevations)
        
        n_pixels = len(image.getdata())
        bbox_area = bounding_box.area
        zones_ratios = {k: v / n_pixels for k, v in zones_data.items()}
        
        return f"Average elevation: {elevations.mean():.2f} meters\n"\
            + f"Max elevation: {elevations.max()} meters\n"\
            + f"Min elevation: {elevations.min()} meters\n\n"\
            + "Elevation zones:\n"\
            + "\n".join([f"{k}: {v * bbox_area:.2f} km squared ({v*100:.2f}%)" for k, v in zones_ratios.items() if v != 0])

def count_elevation_zones(elevation_array):
    zone_counts = {name: 0 for _, _, name in elevation_ranges}  # Initialize counts

    for min_val, max_val, zone_name in elevation_ranges:
        # Count values in the current range
        count = np.sum((elevation_array >= min_val) & (elevation_array < max_val))
        zone_counts[zone_name] += count

    return zone_counts