from datetime import datetime
from typing import Optional, Type
import requests

import numpy as np
import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from tools.input_schemas.temperature_schemas import TemperatureAnalysisInput, TemeperatureForecastInput
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
    

class TemperatureForecastTool(BaseTool):
    name: str = "predict_temperature"
    description: str = "Predict daily minimum, maximum, and mean temperatures for a selected area starting from today's date. Provide forecasts for up to 16 days."
    args_schema: Optional[Type[BaseModel]] = TemeperatureForecastInput

    def _run(self, bounding_box: BoundingBox, forecast_days: int):
        api_url = "https://api.open-meteo.com/v1/forecast"

        center = bounding_box.center

        forecast_days = 16 if forecast_days > 16 else forecast_days
        params = {
            "latitude": center.y,
            "longitude": center.x,
            "current": "temperature_2m",
            "hourly": "temperature_2m",
            "forecast_days": forecast_days,
            "timezone": "UTC",
        }

        response = requests.get(api_url, params=params)

        if forecast_days == 0:
            current_data = response.json()['current']
            formatted_time = datetime.strptime(current_data['time'], '%Y-%m-%dT%H:%M').strftime('%Y-%m-%d')
            return f"Current temperature ({formatted_time}): {current_data['temperature_2m']:.2f} °C"

        df = pd.DataFrame(response.json()["hourly"])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df_daily = df.resample('D').agg({
            'temperature_2m': ['min', 'max', 'mean']
        })
        df_daily.columns = ['daily_min', 'daily_max', 'daily_mean']

        return f"Temperature predictions for the next {forecast_days} days, including today:\n"\
            + '\n'.join(
                f"{dtime.strftime('%Y-%m-%d')}: min: {row['daily_min']:.2f} °C, max: {row['daily_max']:.2f} °C, mean: {row['daily_mean']:.2f} °C"
                for dtime, row in df_daily.iterrows()
            )