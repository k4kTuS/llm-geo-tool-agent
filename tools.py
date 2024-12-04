import datetime

import numpy as np
import pandas as pd
import requests
from statsmodels.tsa.arima.model import ARIMA
from typing_extensions import Annotated, Optional

from darts import TimeSeries
from darts.models import NHiTSModel
from darts.dataprocessing.transformers import Scaler

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

import hotels_model
from config import rgb_LU_mapping
from tool_utils import *

# Tools
@tool(parse_docstring=True)
def get_open_land_use(
    coords: Annotated[list[float], InjectedState("coords")]
) -> str:
    """Return processed open land use data. The bounding box coordinates will be provided during runtime.
    
    Args:
        coords: Map bounding box coordinates.
    """
    image = get_map(coords, "OLU_EU")
    
    n_pixels = len(image.getdata())
    rgb_counts = get_color_counts(image)
    
    land_uses = [rgb_LU_mapping[rgb] for rgb,_ in rgb_counts]
    land_ratios = [cnt/n_pixels for _,cnt in rgb_counts]

    bbox_area = get_area_gpd(coords)
    unit = "km squared"
    if bbox_area < 1:
        bbox_area *= 1000_000
        unit = "m squared"
    
    return f"Map Area: {bbox_area:.2f} {unit}\n\n"\
        + "Land use information for biggest zones:\n"\
        + "\n".join([f"{lu} - Area: {ratio*bbox_area:.2f} {unit} ({ratio*100:.2f}%)" for lu, ratio in zip(land_uses, land_ratios) if ratio > 0.05])

@tool(parse_docstring=True)
def get_monthly_average_temperature_last_5yrs(
    month: str,
    coords: Annotated[list[float], InjectedState("coords")]
) -> str:
    """Return monthly average temperature data calculated from the last five years. The bounding box coordinates will be provided during runtime.
    
    Args:
        month: Month in the format of MM
        coords: Map bounding box coordinates.
    """
    image = get_map(coords, "climate_era5_temperature_last_5yrs_month_avg", {"TIME": f"2020{month}01"})
    
    month_name = datetime.datetime.strptime(month, "%m").strftime("%B")
    return f"Average temperature in {month_name}: {np.array(image).mean():.2f} °C"

@tool(parse_docstring=True)
def get_monthly_average_temperature_prediction_2050s(
    month: str,
    coords: Annotated[list[float], InjectedState("coords")]
) -> str:
    """Return long term forecast of monthly average temperature viable for 2050s.
    The bounding box coordinates will be provided during runtime.
    
    Args:
        month: Month in the format of MM
        coords: Map bounding box coordinates.
    """
    image = get_map(coords, "climate_ipcc_rcp45_temperature_2050s_month_avg", {"TIME": f"2030{month}01"})
    
    month_name = datetime.datetime.strptime(month, "%m").strftime("%B")
    return f"Predicted average temperature in {month_name} in 2050s: {np.array(image).mean():.2f} °C"

@tool(parse_docstring=True)
def get_elevation_data(
    coords: Annotated[list[float], InjectedState("coords")]
) -> str:
    """Return processed data from digital elevation model. The bounding box coordinates will be provided during runtime.
    
    Args:
        coords: Map bounding box coordinates.
    """
    image = get_map(coords, "DEM_MASL")
    elevations = np.array(image)
    zones_data = count_elevation_zones(elevations)
    
    n_pixels = len(image.getdata())
    bbox_area = get_area_gpd(coords)
    zones_ratios = {k: v / n_pixels for k, v in zones_data.items()}
    
    return f"Average elevation: {elevations.mean():.2f} meters\n"\
        + f"Max elevation: {elevations.max()} meters\n"\
        + f"Min elevation: {elevations.min()} meters\n\n"\
        + "Elevation zones:\n"\
        + "\n".join([f"{k}: {v * bbox_area:.2f} km squared ({v*100:.2f}%)" for k, v in zones_ratios.items() if v != 0])

@tool(parse_docstring=True)
def get_eurostat_population_data(
    coords: Annotated[list[float], InjectedState("coords")]
) -> str:
    """Return processed eurostat data about total population. The bounding box coordinates will be provided during runtime.
    
    Args:
        coords: Map bounding box coordinates.
    """
    image = get_map(coords, "EUROSTAT_2021", {"layer": "total_population_eurostat_griddata_2021"})
    total_population = int(np.sum(np.unique(np.array(image))))
    
    return f"Eurostat - Total population: {total_population}"

@tool(parse_docstring=True)
def estimate_hotel_suitability(
    hotel_site_marker: Annotated[Optional[list[float]], InjectedState("hotel_site_marker")]
) -> str:
    """Using data about hotels and other establishments, estimate the number of hotels that could be suitable
    for the marked site. The site marker will be provided during runtime.
    
    Args:
        hotel_site_marker: Highlighted hotel site coordinates.
    """
    if hotel_site_marker is None:
        return "No hotel site marker specified."

    features = hotels_model.load_features()
    model = hotels_model.load_model()

    square_list = features.index.tolist()[1:]
    site_square = find_square_for_marker(square_list, hotel_site_marker)
    if site_square is None:
        return "There is no available data for the marked site."

    square_features = features.loc[site_square]
    return f"Estimated number of hotels suitable for marked site: {model.predict(square_features):.2f}"

@tool(parse_docstring=True)
def predict_temperature(
    steps: int,
    coords: Annotated[list[float], InjectedState("coords")]
) -> str:
    """Predict the temperature of a selected area starting from today's date. Can handle even longer periods of time.
    The bounding box coordinates will be provided during runtime.
    
    Args:
        steps: How long ahead should the prediction be, defined in days. 0 means getting temperature for today.
        coords: Map bounding box coordinates used to calculate the center of the area used for prediction.
    """
    api_url = "https://api.open-meteo.com/v1/forecast"

    lat, lon = [coords[0] + (coords[2] - coords[0]) / 2, coords[1] + (coords[3] - coords[1]) / 2]

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m",
        "hourly": "temperature_2m",
        "past_days": 92,
        "forecast_days": 0,
        "timezone": "UTC",
    }

    response = requests.get(api_url, params=params)

    if steps == 0:
        return f"Current temperature: {response.json()['current']['temperature_2m']:.2f} °C"

    df = pd.DataFrame(response.json()["hourly"])
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df_daily = df.resample('D').mean()['temperature_2m']

    model = ARIMA(df_daily, order=(1,0,0), trend='n')
    model_fit = model.fit()
    preds = model_fit.forecast(steps=steps)

    return f"Predicted temperatures for the next {steps} days:\n" + "\n".join([f"{pred:.2f} °C" for pred in preds])

@tool(parse_docstring=True)
def get_smart_points_of_interest(
    coords: Annotated[list[float], InjectedState("coords")]
) -> str:
    """Return processed data about points of interest in the selected area. The bounding box coordinates will be provided during runtime.
    
    Args:
        coords: Map bounding box coordinates.
    """
    spoi_data = get_spoi_data(coords)
    return f"Number of points of interest: {len(spoi_data['features'])}"\
        + "\n\n" + "\n".join([f"{poi['properties']['cat'][str.rfind(poi['properties']['cat'], '#')+1:]} - {poi['properties']['label']}" for poi in spoi_data['features']])
    
@tool(parse_docstring=True)
def get_tourism_potential(
    years: int,
    coords: Annotated[list[float], InjectedState("coords")]
) -> str:
    """Return historical tourism data together with tourism prediction for a number of years specified in the parameters.
    The bounding box coordinates will be provided during runtime. Prediction can be done for 10 years at most.

    Args:
       years: Number of years to predict tourism in the area for. If 0, only historical data is returned.
       coords: Map bounding box coordinates. 
    """
    data, region_name = get_region_tourism_data(coords)
    
    if data is None:
        if region_name is None:
            return "There is no existing tourism data for the selected region"
        return f"There is no existing tourism data for region {region_name}"

    pds = pd.DataFrame.from_dict(data, orient="index").loc[:, 'all_guests'].astype(float).astype(int)
    pds.index = pds.index.astype(int)

    tourism_data_string = f"Tourism data for region {region_name}:\n\n"\
        + "Number of all guests for recent years:\n"\
        + "\n".join([f"{k}: {v}" for k,v in pds.items()])
    if (years == 0):
        return tourism_data_string
    if (len(pds) < 5):
        return tourism_data_string + "\n\n"\
            + "There is not enough historical tourism data to make a reliable prediction for next years."

    series = TimeSeries.from_series(pds)

    scaler = Scaler()
    series = scaler.fit_transform(series)

    model = NHiTSModel(
        input_chunk_length=4, 
        output_chunk_length=1, 
        n_epochs=20, 
        batch_size=16, 
    )
    model.fit(series)

    forecast = model.predict(n=years)
    forecast = scaler.inverse_transform(forecast)

    last_year = int(list(data.keys())[-1])
    forecast_strings = [
        f"{last_year + i + 1}: {int(v)}" 
        for i, v in enumerate(forecast.values())
    ]
    return tourism_data_string + "\n\n"\
        + f"Prediction for next {years} years:\n"\
        + "\n".join(forecast_strings)

def get_all_tools():
    return [
        get_open_land_use,
        get_monthly_average_temperature_last_5yrs,
        get_monthly_average_temperature_prediction_2050s,
        get_elevation_data,
        get_eurostat_population_data,
        estimate_hotel_suitability,
        predict_temperature,
        get_smart_points_of_interest,
        get_tourism_potential
    ]