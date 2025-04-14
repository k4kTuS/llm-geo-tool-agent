from io import BytesIO
from typing import Optional, Type

import numpy as np
from langchain_core.tools import BaseTool
from PIL import Image
from pydantic import BaseModel
from rasterio.io import MemoryFile
from shapely import box
from typing import Optional, Type

from tools.base_tools import GeospatialTool
from tools.input_schemas.base_schemas import BaseGeomInput
from schemas.geometry import BoundingBox
from utils.tool_utils import get_map_data, get_color_counts
from utils.map_service_utils import LC_rgb_mapping, LU_rgb_mapping, rgb_LC_mapping, rgb_LU_mapping, elevation_ranges


class LandCoverTool(GeospatialTool):
    name: str = "land_cover_tool"
    description: str = (
        "Provides processed land cover data for the bounding box. "
        "Zone types are based on physical surface characteristics."
        "The result is a summary of land cover types and their respective areas."
    )
    args_schema: Optional[Type[BaseModel]] = BaseGeomInput
    boundary = box(-20.0, 30.0, 40.0, 70.0)

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


class LandUseTool(GeospatialTool):
    name: str = "land_use_tool"
    description: str = (
        "Provides processed land use data for the bounding box. "
        "Zone types are based on human activities and planning. "
        "The result is a summary of land use types and their respective areas."
    )
    args_schema: Optional[Type[BaseModel]] = BaseGeomInput
    boundary = box(-20.0, 30.0, 40.0, 70.0)

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