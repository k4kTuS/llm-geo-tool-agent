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
from utils.map_analysis import get_map_data


class TemperatureAnalysisTool(GeospatialTool):
    name: str = "get_monthly_average_temperature_last_5yrs"
    description: str = (
        "Provides monthly average temperature data for the bounding box. "
        "Returns the mean temperature calculated from ERA5 climate reanalysis data averaged over the last five years. "
        "Month is a required parameter, and should be provided as an integer (1-12)."
    )
    args_schema: Optional[Type[BaseModel]] = TemperatureAnalysisInput
    boundary = box(-26.0, 33.9, 32.1, 74.0)

    def _run(self, bounding_box: BoundingBox, month: int) -> str:
        if month < 1 or month > 12:
            raise ValueError("Month must be an integer between 1 and 12.")

        map_data = get_map_data(bounding_box, "climate_era5_temperature_last_5yrs_month_avg", {"TIME": f"2020{month:02d}01"})
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
        month_name = datetime.strptime(str(month), "%m").strftime("%B")
        return f"Average temperature for {month_name} over the last 5 years: {stats['mean']:.2f} °C"


class TemperatureLongPredictionTool(GeospatialTool):
    name: str = "get_monthly_average_temperature_prediction_2030s"
    description: str = (
        "Provides long term monthly temperature prediction for the bounding box. "
        "Predictions are based on the IPCC RCP4.5 climate scenario for the 2030s, calculated as the average "
        "temperature over the period from 2021 to 2040. "
        "Month is a required parameter, and should be provided as an integer (1-12)."
    )
    args_schema: Optional[Type[BaseModel]] = TemperatureAnalysisInput

    def _run(self, bounding_box: BoundingBox, month: int) -> str:
        if month < 1 or month > 12:
            raise ValueError("Month must be an integer between 1 and 12.")

        map_data = get_map_data(bounding_box, "climate_ipcc_rcp45_temperature_2030s_month_avg", {"TIME": f"2030{month:02d}01"})
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
        month_name = datetime.strptime(str(month), "%m").strftime("%B")
        return f"Average temperature according to RCP 4.5 scenario for {month_name} in the 2030s: {stats['mean']:.2f} °C"