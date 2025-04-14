from io import BytesIO
from datetime import datetime
from typing import Optional, Type

import numpy as np
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from rasterio.io import MemoryFile
from shapely.geometry import box

from tools.base import GeospatialTool
from tools.input_schemas.temperature import TemperatureAnalysisInput
from schemas.geometry import BoundingBox
from utils.tool import get_map_data


class TemperatureAnalysisTool(GeospatialTool):
    name: str = "get_monthly_average_temperature_last_5yrs"
    description: str = (
        "Provides monthly average temperature data for the bounding box. "
        "Data is calculated as an average over the last five years from the ERA5 dataset."
    )
    args_schema: Optional[Type[BaseModel]] = TemperatureAnalysisInput
    boundary = box(-26.0, 33.9, 32.1, 74.0)

    def _run(self, bounding_box: BoundingBox, month: str) -> str:
        map_data = get_map_data(bounding_box, "climate_era5_temperature_last_5yrs_month_avg", {"TIME": f"2020{month}01"})
        with MemoryFile(map_data) as memfile:
            with memfile.open() as dataset:
                raster = dataset.read(1)
                # Mask nodata values
                nodata_value = dataset.nodata
                if nodata_value is not None:
                    raster = np.ma.masked_equal(raster, nodata_value)
                stats = {
                    "mean": raster.mean(),
                }
        month_name = datetime.strptime(month, "%m").strftime("%B")
        return f"Average temperature for {month_name} over the last 5 years: {stats['mean']:.2f} °C"


class TemperatureLongPredictionTool(GeospatialTool):
    name: str = "get_monthly_average_temperature_prediction_2030s"
    description: str = (
        "Provides long term monthly temperature prediction for the bounding box."
        "Prediction is based on the IPCC RCP45 scenario for the 2030s calculated by averaging the years 2021 to 2040."
    )
    args_schema: Optional[Type[BaseModel]] = TemperatureAnalysisInput

    def _run(self, bounding_box: BoundingBox, month: str) -> str:
        map_data = get_map_data(bounding_box, "climate_ipcc_rcp45_temperature_2030s_month_avg", {"TIME": f"2030{month}01"})
        with MemoryFile(map_data) as memfile:
            with memfile.open() as dataset:
                raster = dataset.read(1)
                # Mask nodata values
                nodata_value = dataset.nodata
                if nodata_value is not None:
                    raster = np.ma.masked_equal(raster, nodata_value)
                stats = {
                    "mean": raster.mean(),
                }
        month_name = datetime.strptime(month, "%m").strftime("%B")
        return f"Average temperature according to RCP 4.5 scenario for {month_name} in the 2030s: {stats['mean']:.2f} °C"