from datetime import datetime
from typing import Optional, Type

import numpy as np
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from tools.input_schemas.temperature_schemas import TemperatureAnalysisInput
from schemas.geometry import BoundingBox
from utils.tool_utils import get_map


class TemperatureAnalysisTool(BaseTool):
    name: str = "get_monthly_average_temperature_last_5yrs"
    description: str = "Get monthly average temperature data calculated from the last five years."
    args_schema: Optional[Type[BaseModel]] = TemperatureAnalysisInput

    def _run(self, bounding_box: BoundingBox, month: str) -> str:
        image = get_map(bounding_box, "climate_era5_temperature_last_5yrs_month_avg", {"TIME": f"2020{month}01"})
        
        month_name = datetime.strptime(month, "%m").strftime("%B")
        return f"Average temperature in {month_name}: {np.array(image).mean():.2f} °C"


class TemperatureLongPredictionTool(BaseTool):
    name: str = "get_monthly_average_temperature_prediction_2050s"
    description: str = "Get long term forecast of monthly average temperature viable for 2050s."
    args_schema: Optional[Type[BaseModel]] = TemperatureAnalysisInput

    def _run(self, bounding_box: BoundingBox, month: str) -> str:
        image = get_map(bounding_box, "climate_ipcc_rcp45_temperature_2050s_month_avg", {"TIME": f"2030{month}01"})
    
        month_name = datetime.strptime(month, "%m").strftime("%B")
        return f"Predicted average temperature in {month_name} in 2050s: {np.array(image).mean():.2f} °C"