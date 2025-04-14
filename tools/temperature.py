from io import BytesIO
from datetime import datetime
from typing import Optional, Type

import numpy as np
from langchain_core.tools import BaseTool
from PIL import Image
from pydantic import BaseModel
from shapely.geometry import box

from tools.base import GeospatialTool
from tools.input_schemas.temperature import TemperatureAnalysisInput
from schemas.geometry import BoundingBox
from utils.tool import get_map_data


class TemperatureAnalysisTool(GeospatialTool):
    name: str = "get_monthly_average_temperature_last_5yrs"
    description: str = (
        "Provides monthly average temperature data for the bounding box. "
        "Data is calculated as an average over the last five years"
    )
    args_schema: Optional[Type[BaseModel]] = TemperatureAnalysisInput
    boundary = box(-26.0, 33.9, 32.1, 74.0)

    def _run(self, bounding_box: BoundingBox, month: str) -> str:
        map_data = get_map_data(bounding_box, "climate_era5_temperature_last_5yrs_month_avg", {"TIME": f"2020{month}01"})
        image = Image.open(BytesIO(map_data))

        month_name = datetime.strptime(month, "%m").strftime("%B")
        return f"Average temperature in {month_name}: {np.array(image).mean():.2f} °C"


class TemperatureLongPredictionTool(GeospatialTool):
    name: str = "get_monthly_average_temperature_prediction_2030s"
    description: str = (
        "Provides long term temperature prediction for the bounding box."
        "Prediction is based on the IPCC RCP45 scenario for the 2030s. "
    )
    args_schema: Optional[Type[BaseModel]] = TemperatureAnalysisInput

    def _run(self, bounding_box: BoundingBox, month: str) -> str:
        map_data = get_map_data(bounding_box, "climate_ipcc_rcp45_temperature_2030s_month_avg", {"TIME": f"2030{month}01"})
        image = Image.open(BytesIO(map_data))
    
        month_name = datetime.strptime(month, "%m").strftime("%B")
        return f"Predicted average temperature in {month_name} in 2030s: {np.array(image).mean():.2f} °C"