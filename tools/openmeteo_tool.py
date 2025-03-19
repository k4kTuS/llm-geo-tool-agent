import openmeteo_requests
import pandas as pd
import numpy as np

from langchain_core.tools import BaseTool
from pydantic import BaseModel
from typing import Optional, Type, Literal

from tools.input_schemas.openmeteo_schemas import OpenmeteoForecastInput
from tools.input_schemas.base_schemas import BaseGeomInput
from schemas.geometry import BoundingBox
from schemas.data import DataResponse

GRID_SIZE = 4
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"

class CurrentWeatherTool(BaseTool):
    name: str = "current_weather"
    description: str = (
        "A tool for retrieving current weather data based on a given bounding box."
        "It supports the following weather variables: temperature, humidity, precipitation, wind speed, gusts and direction, soil temperature and moisture"
    )
    args_schema: Optional[Type[BaseModel]] = BaseGeomInput

    def _run(self, bounding_box: BoundingBox):
        lat1, lon1, lat2, lon2 = bounding_box.bounds_latlon()

        lat_grid = np.linspace(lat1, lat2, GRID_SIZE)
        lon_grid = np.linspace(lon1, lon2, GRID_SIZE)
        grid_points = [(lat, lon) for lat in lat_grid for lon in lon_grid]

        current_data = get_current_data(grid_points)
        return f"Current weather data for the selected region:\n"\
            + current_data

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
        # Temperature
        temperature_2m_mean = ("temperature_2m", "mean"),
        temperature_2m_min = ("temperature_2m", "min"),
        temperature_2m_max = ("temperature_2m", "max"),
        temperature_2m_std = ("temperature_2m", "std"),
        # Humidity
        relative_humidity_2m_mean = ("relative_humidity_2m", "mean"),
        relative_humidity_2m_std = ("relative_humidity_2m", "std"),
        # Precipitation
        precipitation_probability_max = ("precipitation_probability", "max"),
        precipitation_total = ("precipitation", "sum"),
        # Wind
        wind_speed_10m_max = ("wind_speed_10m", "max"),
        wind_gusts_10m_max = ("wind_gusts_10m", "max"),
        wind_direction_10m_list = ("wind_direction_10m", lambda x: [round(val, 2) for val in x]),
        wind_speed_10m_list = ("wind_speed_10m", lambda x: [round(val, 2) for val in x]),
        # Soil Temperature
        soil_temperature_0cm_mean = ("soil_temperature_0cm", "mean"),
        soil_temperature_0cm_max = ("soil_temperature_0cm", "min"),
        soil_temperature_0cm_min = ("soil_temperature_0cm", "max"),
        # Soil Moisture
        soil_moisture_0_to_1cm_mean = ("soil_moisture_0_to_1cm", "mean"),
        soil_moisture_0_to_1cm_max = ("soil_moisture_0_to_1cm", "max"),
        soil_moisture_0_to_1cm_min = ("soil_moisture_0_to_1cm", "min"),
    ).reset_index()

    hourly_area_summary.columns = [
        'date',
        'temperature_2m_mean (°C)',
        'temperature_2m_min (°C)',
        'temperature_2m_max (°C)',
        'temperature_2m_std (°C)',
        'relative_humidity_2m_mean (%)',
        'relative_humidity_2m_std (%)',
        'precipitation_probability_max (%)',
        'precipitation_total (mm)',
        'wind_speed_10m_max (km/h)',
        'wind_gusts_10m_max (km/h)',
        'wind_direction_10m_measurements (°)',
        'wind_speed_10m_measurements (km/h)',
        'soil_temperature_0cm_mean (°C)',
        'soil_temperature_0cm_max (°C)',
        'soil_temperature_0cm_min (°C)',
        'soil_moisture_0_to_1cm_mean (m³/m³)',
        'soil_moisture_0_to_1cm_max (m³/m³)',
        'soil_moisture_0_to_1cm_min (m³/m³)'
    ]

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
        wind_direction_10m_dominant_list = ("wind_direction_10m_dominant",  lambda x: [round(val, 2) for val in x])
    ).reset_index()

    daily_area_summary.columns = [
        'date',
        'temperature_2m_max (°C)',
        'temperature_2m_min (°C)',
        'daylight_duration (s)',
        'sunshine_duration (s)',
        'precipitation_sum (mm)',
        'precipitation_hours (h)',
        'precipitation_probability_max (%)',
        'wind_speed_10m_max (km/h)',
        'wind_gusts_10m_max (km/h)',
        'wind_direction_10m_dominant_measurements (°)'
    ]

    return daily_area_summary

def get_current_data(grid_points) -> str:
    params = {
        "latitude": [lat for lat, _ in grid_points],
        "longitude": [lon for _, lon in grid_points],
        "current": ["temperature_2m", "relative_humidity_2m", "precipitation_probability", "precipitation", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "soil_temperature_0cm", "soil_moisture_0_to_1cm"],
        "timezone": "UTC"
    }

    openmeteo = openmeteo_requests.Client()
    responses = openmeteo.weather_api(OPENMETEO_URL, params=params)

    data = {
        "temperature_2m": [],
        "relative_humidity_2m": [],
        "precipitation_probability": [],
        "precipitation": [],
        "wind_speed_10m": [],
        "wind_direction_10m": [],
        "wind_gusts_10m": [],
        "soil_temperature_0cm": [],
        "soil_moisture_0_to_1cm": [],
    }
    # Process all locations
    for response in responses:
        # Process current data. The order of variables needs to be the same as requested.
        current = response.Current()
        data["temperature_2m"].append(current.Variables(0).Value())
        data["relative_humidity_2m"].append(current.Variables(1).Value())
        data["precipitation_probability"].append(current.Variables(2).Value())
        data["precipitation"].append(current.Variables(3).Value())
        data["wind_speed_10m"].append(current.Variables(4).Value())
        data["wind_direction_10m"].append(current.Variables(5).Value())
        data["wind_gusts_10m"].append(current.Variables(6).Value())
        data["soil_temperature_0cm"].append(current.Variables(7).Value())
        data["soil_moisture_0_to_1cm"].append(current.Variables(8).Value())

    # Define aggregation mapping
    aggregated_data = {
        # Temperature
        "temperature_2m_mean": np.mean(data["temperature_2m"]),
        "temperature_2m_min": np.min(data["temperature_2m"]),
        "temperature_2m_max": np.max(data["temperature_2m"]),
        "temperature_2m_std": np.std(data["temperature_2m"]),
        # Humidity
        "relative_humidity_2m_mean": np.mean(data["relative_humidity_2m"]),
        "relative_humidity_2m_std": np.std(data["relative_humidity_2m"]),
        # Precipitation
        "precipitation_probability_max": np.max(data["precipitation_probability"]),
        "precipitation_total": np.sum(data["precipitation"]),
        # Wind
        "wind_speed_10m_max": np.max(data["wind_speed_10m"]),
        "wind_gusts_10m_max": np.max(data["wind_gusts_10m"]),
        "wind_speed_10m_all": data["wind_speed_10m"],
        "wind_direction_10m_all": data["wind_direction_10m"],
        # Soil Temperature
        "soil_temperature_0cm_mean": np.mean(data["soil_temperature_0cm"]),
        "soil_temperature_0cm_min": np.min(data["soil_temperature_0cm"]),
        "soil_temperature_0cm_max": np.max(data["soil_temperature_0cm"]),
        # Soil Moisture
        "soil_moisture_0_to_1cm_mean": np.mean(data["soil_moisture_0_to_1cm"]),
        "soil_moisture_0_to_1cm_min": np.min(data["soil_moisture_0_to_1cm"]),
        "soil_moisture_0_to_1cm_max": np.max(data["soil_moisture_0_to_1cm"]),
    }

    text_summary = (
        f"**Temperature:** Avg: {aggregated_data['temperature_2m_mean']:.1f}°C, "
        f"Min: {aggregated_data['temperature_2m_min']:.1f}°C, Max: {aggregated_data['temperature_2m_max']:.1f}°C "
        f"(Std Dev: {aggregated_data['temperature_2m_std']:.1f}°C)\n"
        
        f"**Humidity:** {aggregated_data['relative_humidity_2m_mean']:.1f}% "
        f"(Std Dev: {aggregated_data['relative_humidity_2m_std']:.1f}%)\n"
        
        f"**Precipitation:** {aggregated_data['precipitation_total']:.1f} mm "
        f"(Max Chance: {aggregated_data['precipitation_probability_max']:.1f}%)\n"
        
        f"**Wind:** Max: {aggregated_data['wind_speed_10m_max']:.1f} km/h with Gusts (Max): {aggregated_data['wind_gusts_10m_max']:.1f} km/h\n"
        f"**All wind measurements:**\n"
        f"{", ".join([f"{speed:.1f} km/h from {direction:.1f}°" for speed, direction in zip(aggregated_data['wind_speed_10m_all'], aggregated_data['wind_direction_10m_all'])])}\n"
        
        f"**Soil Temperature:** Avg: {aggregated_data['soil_temperature_0cm_mean']:.1f}°C, "
        f"Min: {aggregated_data['soil_temperature_0cm_min']:.1f}°C, Max: {aggregated_data['soil_temperature_0cm_max']:.1f}°C\n"
        
        f"**Soil Moisture:** Avg: {aggregated_data['soil_moisture_0_to_1cm_mean']:.2f} m³/m³, "
        f"Min: {aggregated_data['soil_moisture_0_to_1cm_min']:.2f} m³/m³, Max: {aggregated_data['soil_moisture_0_to_1cm_max']:.2f} m³/m³"
    )
    return text_summary