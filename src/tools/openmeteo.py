import openmeteo_requests
import pandas as pd
import numpy as np
from datetime import datetime as dt

from langchain_core.tools import BaseTool
from pydantic import BaseModel
from typing import Optional, Type, Literal

from tools.base import GeospatialTool
from tools.input_schemas.openmeteo import OpenmeteoForecastInput
from tools.input_schemas.base import BaseGeomInput
from schemas.geometry import BoundingBox
from schemas.data import DataResponse
from utils.openmeteo import generate_grid_points, describe_dominant_wind_direction

OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_DAYS_MAX = 7

class CurrentWeatherTool(GeospatialTool):
    name: str = "current_weather"
    description: str = (
        "Retrieves current weather data for the bounding box. "
        "You'll recieve these weather variables: temperature, humidity, precipitation, wind speed, wind gusts, wind direction, soil temperature, soil moisture"
    )
    args_schema: Optional[Type[BaseModel]] = BaseGeomInput
    response_format: str = "content_and_artifact"

    def _run(self, bounding_box: BoundingBox):
        current_data = get_current_data(generate_grid_points(bounding_box))
        output = f"Current weather data for the selected region:\n" + current_data
        data_output = DataResponse(
            name="Current weather data",
            source="OpenMeteo",
            data_type="text",
            data=current_data,
            show_data=True
        )
        return output, data_output

class WeatherForecastTool(GeospatialTool):
    name: str = "weather_forecast"
    description: str = (
        "Retrieves weather forecast data for the bounding box. "
        f"Provides forecasts for up to {FORECAST_DAYS_MAX} days ahead - no historical data available."
        "\nYou can request either:"
        "\n- 'hourly' forecasts: giving detailed hour-by-hour predictions"
        "\n- 'daily' forecasts: providing day-level summaries"
        "\n\nFor hourly forecasts, you'll receive: temperature, humidity, precipitation, wind speed, wind gusts, wind direction, soil temperature, soil moisture."
        "\nFor daily forecasts, you'll receive: max temperature, min temperature, daylight duration, sunshine duration, precipitation, precipitation hours, wind speed, wind gusts, dominant wind direction."
    )
    args_schema: Optional[Type[BaseModel]] = OpenmeteoForecastInput
    response_format: str = "content_and_artifact"

    def _run(self, bounding_box: BoundingBox, date_start, date_end, forecast_type: Literal["hourly", "daily"]):
        try:
            start_date = dt.strptime(date_start, "%Y-%m-%d")
        except ValueError as e:
            return "ValueError: Invalid start date format. Please use YYYY-MM-DD.", None
        try:
            end_date = dt.strptime(date_end, "%Y-%m-%d")
        except ValueError as e:
            return "ValueError: Invalid end date format. Please use YYYY-MM-DD.", None
        if end_date < start_date:
            return "Error: End date must be after start date.", None
        if (start_date.date() < dt.now().date()):
            return "Error: Could not provide historical weather data.", None
        
        date_range_str = f"for {start_date.strftime("%Y-%m-%d")}" if start_date == end_date else f"from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}"
        if (end_date - dt.now()).days > FORECAST_DAYS_MAX:
            return f"Could not provide weather data {date_range_str}. Data is available only for up to {FORECAST_DAYS_MAX} days from today, including today.", None

        grid_points = generate_grid_points(bounding_box)

        if forecast_type == "hourly":
            hourly_data = get_hourly_data(grid_points, start_date, end_date)
            text_output = f"Hourly weather data {date_range_str}:\n"\
                + hourly_data.to_markdown(index=False)
            data_output = DataResponse(
                name="Weather forecast data",
                source="OpenMeteo",
                data_type="dataframe",
                data=hourly_data,
                show_data=True
            )   
        elif forecast_type == "daily":
            daily_data = get_daily_data(grid_points, start_date, end_date)
            text_output =  f"Daily weather data {date_range_str}:\n"\
                + daily_data.to_markdown(index=False)
            data_output = DataResponse(
                name="Weather forecast data",
                source="OpenMeteo",
                data_type="dataframe",
                data=daily_data,
                show_data=True
            )
        else:
            return "Error: Invalid forecast type. Please use 'hourly' or 'daily'.", None
        return text_output, data_output

def get_hourly_data(grid_points, start_date: dt, end_date: dt) -> pd.DataFrame:
    params = {
        "latitude": [lat for lat, _ in grid_points],
        "longitude": [lon for _, lon in grid_points],
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "soil_temperature_0cm", "soil_moisture_0_to_1cm"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "UTC",
        "models": "icon_seamless"
    }

    openmeteo = openmeteo_requests.Client()
    responses = openmeteo.weather_api(OPENMETEO_URL, params=params)

    all_df = pd.DataFrame()
    # Process all locations
    for response in responses:

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()

        hourly_data = {"date": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}

        hourly_data["temperature_2m"] = hourly.Variables(0).ValuesAsNumpy()
        hourly_data["relative_humidity_2m"] = hourly.Variables(1).ValuesAsNumpy()
        hourly_data["precipitation"] = hourly.Variables(2).ValuesAsNumpy()
        hourly_data["wind_speed_10m"] = hourly.Variables(3).ValuesAsNumpy()
        hourly_data["wind_direction_10m"] = hourly.Variables(4).ValuesAsNumpy()
        hourly_data["wind_gusts_10m"] = hourly.Variables(5).ValuesAsNumpy()
        hourly_data["soil_temperature_0cm"] = hourly.Variables(6).ValuesAsNumpy()
        hourly_data["soil_moisture_0_to_1cm"] = hourly.Variables(7).ValuesAsNumpy()

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
        precipitation_total = ("precipitation", "sum"),
        # Wind
        wind_speed_10m_max = ("wind_speed_10m", "max"),
        wind_gusts_10m_max = ("wind_gusts_10m", "max"),
        wind_direction_10m_dominant = ("wind_direction_10m", lambda x: describe_dominant_wind_direction([val for val in x])),
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
        'precipitation_total (mm)',
        'wind_speed_10m_max (km/h)',
        'wind_gusts_10m_max (km/h)',
        'dominant_wind_direction_10m (from all measurements)',
        'soil_temperature_0cm_mean (°C)',
        'soil_temperature_0cm_max (°C)',
        'soil_temperature_0cm_min (°C)',
        'soil_moisture_0_to_1cm_mean (m³/m³)',
        'soil_moisture_0_to_1cm_max (m³/m³)',
        'soil_moisture_0_to_1cm_min (m³/m³)'
    ]

    return hourly_area_summary

def get_daily_data(grid_points, start_date: dt, end_date: dt) -> pd.DataFrame:
    params = {
        "latitude": [lat for lat, _ in grid_points],
        "longitude": [lon for _, lon in grid_points],
        "daily": ["temperature_2m_max", "temperature_2m_min", "daylight_duration", "sunshine_duration", "precipitation_sum", "precipitation_hours", "precipitation_probability_max", "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "UTC",
        "models": "icon_seamless"
    }

    openmeteo = openmeteo_requests.Client()
    responses = openmeteo.weather_api(OPENMETEO_URL, params=params)

    all_df = pd.DataFrame()

    for response in responses:
        daily = response.Daily()

        daily_data = {"date": pd.date_range(
            start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
            end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = daily.Interval()),
            inclusive = "left"
        )}

        daily_data["temperature_2m_max"] = daily.Variables(0).ValuesAsNumpy()
        daily_data["temperature_2m_min"] = daily.Variables(1).ValuesAsNumpy()
        daily_data["daylight_duration"] = daily.Variables(2).ValuesAsNumpy()
        daily_data["sunshine_duration"] = daily.Variables(3).ValuesAsNumpy()
        daily_data["precipitation_sum"] = daily.Variables(4).ValuesAsNumpy()
        daily_data["precipitation_hours"] = daily.Variables(5).ValuesAsNumpy()
        daily_data["precipitation_probability_max"] = daily.Variables(6).ValuesAsNumpy()
        daily_data["wind_speed_10m_max"] = daily.Variables(7).ValuesAsNumpy()
        daily_data["wind_gusts_10m_max"] = daily.Variables(8).ValuesAsNumpy()
        daily_data["wind_direction_10m_dominant"] = daily.Variables(9).ValuesAsNumpy()

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
        wind_direction_10m_dominant = ("wind_direction_10m_dominant", lambda x: describe_dominant_wind_direction([val for val in x])),
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
        'wind_direction_10m_dominant (from all measurements)',
    ]

    return daily_area_summary

def get_current_data(grid_points) -> str:
    params = {
        "latitude": [lat for lat, _ in grid_points],
        "longitude": [lon for _, lon in grid_points],
        "current": ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "soil_temperature_0cm", "soil_moisture_0_to_1cm"],
        "timezone": "UTC"
    }

    openmeteo = openmeteo_requests.Client()
    responses = openmeteo.weather_api(OPENMETEO_URL, params=params)

    data = {
        "temperature_2m": [],
        "relative_humidity_2m": [],
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
        data["precipitation"].append(current.Variables(2).Value())
        data["wind_speed_10m"].append(current.Variables(3).Value())
        data["wind_direction_10m"].append(current.Variables(4).Value())
        data["wind_gusts_10m"].append(current.Variables(5).Value())
        data["soil_temperature_0cm"].append(current.Variables(6).Value())
        data["soil_moisture_0_to_1cm"].append(current.Variables(7).Value())

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
        "precipitation_total": np.sum(data["precipitation"]),
        # Wind
        "wind_speed_10m_max": np.max(data["wind_speed_10m"]),
        "wind_gusts_10m_max": np.max(data["wind_gusts_10m"]),
        "wind_direction_10m_dominant": describe_dominant_wind_direction(data["wind_direction_10m"]),
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
        
        f"**Precipitation:** {aggregated_data['precipitation_total']:.1f} mm\n"
        
        f"**Wind:** Max: {aggregated_data['wind_speed_10m_max']:.1f} km/h with Gusts (Max): {aggregated_data['wind_gusts_10m_max']:.1f} km/h\n"
        f"**Dominant wind directions (from all measurements):**\n"
        f"{aggregated_data['wind_direction_10m_dominant']}\n"
        
        f"**Soil Temperature:** Avg: {aggregated_data['soil_temperature_0cm_mean']:.1f}°C, "
        f"Min: {aggregated_data['soil_temperature_0cm_min']:.1f}°C, Max: {aggregated_data['soil_temperature_0cm_max']:.1f}°C\n"
        
        f"**Soil Moisture:** Avg: {aggregated_data['soil_moisture_0_to_1cm_mean']:.2f} m³/m³, "
        f"Min: {aggregated_data['soil_moisture_0_to_1cm_min']:.2f} m³/m³, Max: {aggregated_data['soil_moisture_0_to_1cm_max']:.2f} m³/m³"
    )
    return text_summary