import openmeteo_requests
import pandas as pd
import numpy as np

from langchain_core.tools import BaseTool
from pydantic import BaseModel
from typing import Optional, Type, Literal

from tools.input_schemas.openmeteo_schemas import OpenmeteoForecastInput
from schemas.geometry import BoundingBox
from schemas.data import DataResponse

GRID_SIZE = 4
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"

class WeatherForecastTool(BaseTool):
    name: str = "weather_forecast"
    description: str = (
        "A tool for retrieving weather forecast data based on a given bounding box. "
        "It supports both hourly and daily forecasts for up to 16 days. "
        "\nReturns the following weather variables:"
        "\nhourly data: temperature, humidity, precipitation, wind speed, gusts and direction, soil temperature and moisture"
        "\ndaily data: max and min temperature, daylight and sunshine duration, precipitation and precipitation hours, wind speed, gusts and dominnat direction"
    )
    args_schema: Optional[Type[BaseModel]] = OpenmeteoForecastInput
    response_format: str = "content_and_artifact"

    def _run(self, bounding_box: BoundingBox, forecast_days: int, forecast_type: Literal["hourly", "daily"]):
        lat1, lon1, lat2, lon2 = bounding_box.bounds_latlon()

        lat_grid = np.linspace(lat1, lat2, GRID_SIZE)
        lon_grid = np.linspace(lon1, lon2, GRID_SIZE)
        grid_points = [(lat, lon) for lat in lat_grid for lon in lon_grid]
        
        # Limit forecast days to 16
        forecast_days = max(1, min(forecast_days, 16))
        if forecast_type == "hourly":
            hourly_data = get_hourly_data(grid_points, forecast_days)
            text_output = f"Hourly weather data for the next {forecast_days} days:\n"\
                + hourly_data.to_markdown(index=False)
            data_output = DataResponse(
                name="Weather forecast data",
                source="OpenMeteo",
                data_type="dataframe",
                data=hourly_data,
                show_data=True
            )   
        elif forecast_type == "daily":
            daily_data = get_daily_data(grid_points, forecast_days)
            text_output =  f"Daily weather data for the next {forecast_days} days:\n"\
                + daily_data.to_markdown(index=False)
            data_output = DataResponse(
                name="Weather forecast data",
                source="OpenMeteo",
                data_type="dataframe",
                data=daily_data,
                show_data=True
            )
        return text_output, data_output

def get_hourly_data(grid_points, forecast_days) -> pd.DataFrame:
    params = {
        "latitude": [lat for lat, _ in grid_points],
        "longitude": [lon for _, lon in grid_points],
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation_probability", "precipitation", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "soil_temperature_0cm", "soil_moisture_0_to_1cm"],
        "forecast_days": forecast_days,
        "timezone": "UTC"
    }

    openmeteo = openmeteo_requests.Client()
    responses = openmeteo.weather_api(OPENMETEO_URL, params=params)

    all_df = pd.DataFrame()
    # Process all locations
    for response in responses:

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
        hourly_precipitation_probability = hourly.Variables(2).ValuesAsNumpy()
        hourly_precipitation = hourly.Variables(3).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(4).ValuesAsNumpy()
        hourly_wind_direction_10m = hourly.Variables(5).ValuesAsNumpy()
        hourly_wind_gusts_10m = hourly.Variables(6).ValuesAsNumpy()
        hourly_soil_temperature_0cm = hourly.Variables(7).ValuesAsNumpy()
        hourly_soil_moisture_0_to_1cm = hourly.Variables(8).ValuesAsNumpy()

        hourly_data = {"date": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}

        hourly_data["temperature_2m"] = hourly_temperature_2m
        hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
        hourly_data["precipitation_probability"] = hourly_precipitation_probability
        hourly_data["precipitation"] = hourly_precipitation
        hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
        hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
        hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m
        hourly_data["soil_temperature_0cm"] = hourly_soil_temperature_0cm
        hourly_data["soil_moisture_0_to_1cm"] = hourly_soil_moisture_0_to_1cm

        hourly_dataframe = pd.DataFrame(data = hourly_data)
        all_df = pd.concat([all_df, hourly_dataframe])

    # Aggregate hourly data from all locations
    hourly_area_summary = all_df.groupby("date").agg(
        temperature_2m = ("temperature_2m", "mean"),
        relative_humidity_2m = ("relative_humidity_2m", "mean"),
        precipitation_probability = ("precipitation_probability", "mean"),
        precipitation = ("precipitation", "sum"),
        wind_speed_10m = ("wind_speed_10m", "mean"),
        wind_direction_10m = ("wind_direction_10m", "mean"),
        wind_gusts_10m = ("wind_gusts_10m", "mean"),
        soil_temperature_0cm = ("soil_temperature_0cm", "mean"),
        soil_moisture_0_to_1cm = ("soil_moisture_0_to_1cm", "mean")
    ).reset_index()

    return hourly_area_summary

def get_daily_data(grid_points, forecast_days) -> pd.DataFrame:
    params = {
        "latitude": [lat for lat, _ in grid_points],
        "longitude": [lon for _, lon in grid_points],
        "daily": ["temperature_2m_max", "temperature_2m_min", "daylight_duration", "sunshine_duration", "precipitation_sum", "precipitation_hours", "precipitation_probability_max", "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant"],
        "forecast_days": forecast_days,
        "timezone": "UTC"
    }

    openmeteo = openmeteo_requests.Client()
    responses = openmeteo.weather_api(OPENMETEO_URL, params=params)

    all_df = pd.DataFrame()

    for response in responses:
        daily = response.Daily()
        daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
        daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
        daily_daylight_duration = daily.Variables(2).ValuesAsNumpy()
        daily_sunshine_duration = daily.Variables(3).ValuesAsNumpy()
        daily_precipitation_sum = daily.Variables(4).ValuesAsNumpy()
        daily_precipitation_hours = daily.Variables(5).ValuesAsNumpy()
        daily_precipitation_probability_max = daily.Variables(6).ValuesAsNumpy()
        daily_wind_speed_10m_max = daily.Variables(7).ValuesAsNumpy()
        daily_wind_gusts_10m_max = daily.Variables(8).ValuesAsNumpy()
        daily_wind_direction_10m_dominant = daily.Variables(9).ValuesAsNumpy()

        daily_data = {"date": pd.date_range(
            start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
            end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = daily.Interval()),
            inclusive = "left"
        )}

        daily_data["temperature_2m_max"] = daily_temperature_2m_max
        daily_data["temperature_2m_min"] = daily_temperature_2m_min
        daily_data["daylight_duration"] = daily_daylight_duration
        daily_data["sunshine_duration"] = daily_sunshine_duration
        daily_data["precipitation_sum"] = daily_precipitation_sum
        daily_data["precipitation_hours"] = daily_precipitation_hours
        daily_data["precipitation_probability_max"] = daily_precipitation_probability_max
        daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
        daily_data["wind_gusts_10m_max"] = daily_wind_gusts_10m_max
        daily_data["wind_direction_10m_dominant"] = daily_wind_direction_10m_dominant

        daily_dataframe = pd.DataFrame(data = daily_data)
        all_df = pd.concat([all_df, daily_dataframe])
    
    # Aggregate daily data from all locations
    daily_area_summary = all_df.groupby("date").agg(
        temperature_2m_max = ("temperature_2m_max", "mean"),
        temperature_2m_min = ("temperature_2m_min", "mean"),
        daylight_duration = ("daylight_duration", "mean"),
        sunshine_duration = ("sunshine_duration", "mean"),
        precipitation_sum = ("precipitation_sum", "sum"),
        precipitation_hours = ("precipitation_hours", "mean"),
        precipitation_probability_max = ("precipitation_probability_max", "mean"),
        wind_speed_10m_max = ("wind_speed_10m_max", "mean"),
        wind_gusts_10m_max = ("wind_gusts_10m_max", "mean"),
        wind_direction_10m_dominant = ("wind_direction_10m_dominant", "mean")
    ).reset_index()

    return daily_area_summary