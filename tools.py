import datetime
import numpy as np
import pandas as pd
import requests

from io import BytesIO
from PIL import Image
from pyproj import Transformer
from shapely.geometry import Polygon

# Tools
from typing_extensions import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

import hotels_model

map_config={
'transport':
    {'wms_root_url':'https://gis.lesprojekt.cz/cgi-bin/mapserv', 
    'data':{'map':'/home/dima/maps/open_transport_map/open_transport_map.map', 'service':'WMS', 'version':'1.3.0', 'request':'GetMap', 'bbox':'49.3,12.7,49.4,12.8', 'crs':'EPSG:4326', 'width':'1562', 'height':'680', 'layers':'road_classes_all', 'styles':'', 'format':'png' }, 
    'alternatives':{}
    }, 
'climate_era5_temperature_last_5yrs_month_avg':
    {'wms_root_url':'https://olu.lesprojekt.cz/cgi-bin/mapserv', 
    'data':{'map':'/data/maps/thematic_maps.map', 'service':'WMS', 'version':'1.3.0', 'request':'GetMap', 'bbox':'49.3,12.7,49.4,12.8', 'crs':'EPSG:4326', 'width':'1562', 'height':'680', 'layers':'t2m_2020', 'TIME':'20200101','styles':'', 'format':'gtiff' }, 
    'alternatives':{'TIME':[datetime.date(2020,i,1).strftime('%Y%m%d') for i in range(1,13)]}
    }, 
'climate_ipcc_rcp45_temperature_2050s_month_avg':
    {'wms_root_url':'https://olu.lesprojekt.cz/cgi-bin/mapserv', 
    'data':{'map':'/data/maps/thematic_maps.map', 'service':'WMS', 'version':'1.3.0', 'request':'GetMap', 'bbox':'49.3,12.7,49.4,12.8', 'crs':'EPSG:4326', 'width':'1562', 'height':'680', 'layers':'tas_2030', 'TIME':'20300101','styles':'', 'format':'gtiff' }, 
    'alternatives':{'TIME':[datetime.date(2030,i,1).strftime('%Y%m%d') for i in range(1,13)]}
    }, 
'OLU_EU':
    {'wms_root_url':'https://olu.lesprojekt.cz/cgi-bin/mapserv', 
    'data':{'map':'/data/maps/olu_europe.map', 'service':'WMS', 'version':'1.3.0', 'request':'GetMap', 'bbox':'49.3,12.7,49.4,12.8', 'crs':'EPSG:4326', 'width':'1562', 'height':'680', 'layers':'olu_obj_lu', 'styles':'', 'format':'png' }, 
    'alternatives':{}
    }, 
'OLU_CZ':
    {'wms_root_url':'https://olu.lesprojekt.cz/cgi-bin/mapserv', 
    'data':{'map':'/data/maps/olu_europe.map', 'service':'WMS', 'version':'1.3.0', 'request':'GetMap', 'bbox':'49.3,12.7,49.4,12.8', 'crs':'EPSG:4326', 'width':'3000', 'height':'3000', 'layers':'olu_bbox_ts', 'styles':'', 'format':'png' }, 
    'alternatives':{'TIME':[datetime.date(i,12,31).strftime('%Y-%m-%d') for i in range(2015,2024)]}
    }, 
'EUROSTAT_2021':
    {'wms_root_url':'https://olu.lesprojekt.cz/cgi-bin/mapserv', 
    'data':{'map':'/data/maps/thematic_maps.map', 'service':'WMS', 'version':'1.3.0', 'request':'GetMap', 'bbox':'49.3,12.7,49.4,12.8', 'crs':'EPSG:4326', 'width':'1562', 'height':'680', 'layers':'total_population_eurostat_griddata_2021', 'styles':'', 'format':'gtiff' }, 
    'alternatives':{'layers':['total_population_eurostat_griddata_2021', 'employed_population_eurostat_griddata_2021']}
    }, 
'DEM_color':
    {'wms_root_url':'https://gis.lesprojekt.cz/cgi-bin/mapserv', 
    'data':{'map':'/home/dima/maps/foodie/dem.map', 'service':'WMS', 'version':'1.3.0', 'request':'GetMap', 'bbox':'49.3,12.7,49.4,12.8', 'crs':'EPSG:4326', 'width':'1562', 'height':'680', 'layers':'DEM', 'styles':'', 'format':'png' }, 
    'alternatives':{}
    }, 
}

# Used with DEM
elevation_ranges = [
    (0, 500, "Lowland"),
    (500, 1500, "Upland"),
    (1500, 2500, "Highland"),
    (2500, 3500, "Alpine"),
    (3500, 4500, "Subnival"),
    (4500, np.inf, "Nival")
]

def count_elevation_zones(elevation_array):
    zone_counts = {name: 0 for _, _, name in elevation_ranges}  # Initialize counts
    
    for min_val, max_val, zone_name in elevation_ranges:
        # Count values in the current range
        count = np.sum((elevation_array >= min_val) & (elevation_array < max_val))
        zone_counts[zone_name] += count
    
    return zone_counts

def rgb2elevation(image):
    # Reference points - Snezka, Labe
    elevation_1, hue_1 = (1603, 14)
    elevation_2, hue_2 = (115, 78)
        
    hue_channel = np.array(image.convert('HSV'))[:, :, 0]

    # Linear interpolation for each hue value
    elevation_masl = ((elevation_2 - elevation_1) / (hue_2 - hue_1)) * (hue_channel - hue_1) + elevation_1

    # Clip values to avoid unrealistic elevations outside of the given hue range
    return np.clip(elevation_masl, elevation_2, elevation_1)

# Used with OLU
# Working only for a hard-coded bounding box, different areas can have other colors
rgb2color_mapping = {
    (240, 120, 100) : 'red',
    (100, 100, 100) : 'darkgrey',
    (230, 230, 110) : 'yellow',
    (180, 120, 240) : 'purple',
    (120, 170, 150) : 'darkgreen',
    (220, 160, 220) : 'lightpurple',
    (110, 230, 110) : 'lightgreen',
    (200, 220, 220) : 'lightblue',
    (220, 220, 220) : 'whitegrey',
    (240, 240, 240) : 'white'
}

color2olu_mapping = {
    'darkgreen': 'Forest Zone',
    'darkgrey': 'Industrial Zone',
    'lightblue': 'Waterbody Zone',
    'lightgreen': 'Park Zone',
    'purple': 'Road Communication Zone',
    'lightpurple': 'Pedestrian Zone',
    'red': 'Commercial Zone',
    'whitegrey': 'Parking Zone',
    'white': 'Utilities Zone',
    'yellow': 'Residential Zone'
}

def get_color_counts(image, n_colors=10):
    pixels = np.array(image)
    pixels = pixels.reshape(-1, 3)

    pixels_view = pixels.view(np.dtype((np.void, pixels.dtype.itemsize * pixels.shape[1])))
    unique_pixels, counts = np.unique(pixels_view, return_counts=True)

    # Convert back to normal RGB tuples
    unique_pixels = unique_pixels.view(pixels.dtype).reshape(-1, 3)

    # Combine pixels and their counts, and sort by count
    sorted_pixels = sorted(zip(map(tuple,unique_pixels), counts), key=lambda x: x[1], reverse=True)

    # Get the top n colors
    return sorted_pixels[:n_colors]

# Other helpers
def get_map(coords, endpoint, alt_params={}):
    bbox = ','.join([str(v) for v in coords])
    
    api_setup = map_config[endpoint]
    response = requests.get(
        api_setup["wms_root_url"],
        params={**api_setup["data"], **{"bbox": bbox, "height":"1500", "width":"1500"}, **alt_params},
        stream=True
    )
    return Image.open(BytesIO(response.content))

def get_area(coords):
    # Initialize a Transformer to convert from WGS84 (EPSG:4326) to a projected system
    # EPSG:4326 is latitude/longitude in degrees, EPSG:3395 is Pseudo Mercator in meters
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    # Transform coordinates - first array containts longitudes, second latitudes
    lng, lat = transformer.transform([coords[1], coords[3]], [coords[0], coords[2]])
    # Create polygon ring
    ring = [(lng[0], lat[0]), (lng[0], lat[1]), (lng[1], lat[1]), (lng[1], lat[0])]
    polygon = Polygon(ring)
    # Polygon area in km2
    return polygon.area/1000000

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
    
    bbox_area = get_area(coords)
    n_pixels = len(image.getdata())
    top_colors = get_color_counts(image, n_colors=10)
    
    land_uses = [color2olu_mapping[rgb2color_mapping[v[0]]] for v in top_colors]
    land_areas = [v[1]/n_pixels * bbox_area for v in top_colors]
    
    return f"Map Area: {bbox_area:.2f} km squared\n\n"\
        + "Land use information:\n"\
        + "\n".join([f"{v[0]} - Area: {v[1]:.2f} km squared" for v in zip(land_uses, land_areas)])

@tool(parse_docstring=True)
def get_monthly_average_temperature(
    month: str,
    coords: Annotated[list[float], InjectedState("coords")]
) -> str:
    """Return processed monthly average temperature data. The bounding box coordinates will be provided during runtime.
    
    Args:
        month: Month in the format of MM
        coords: Map bounding box coordinates.
    """
    image = get_map(coords, "climate_era5_temperature_last_5yrs_month_avg", {"TIME": f"2020{month}01"})
    
    month_name = datetime.datetime.strptime(month, "%m").strftime("%B")
    return f"Average temperature in {month_name}: {np.array(image).mean():.2f} degrees Celsius."

@tool(parse_docstring=True)
def get_monthly_average_temperature_prediction(
    month: str,
    coords: Annotated[list[float], InjectedState("coords")]
) -> str:
    """Return processed prediction of monthly average temperature data in year 2030. The bounding box coordinates will be provided during runtime.
    
    Args:
        month: Month in the format of MM
        coords: Map bounding box coordinates.
    """
    image = get_map(coords, "climate_ipcc_rcp45_temperature_2050s_month_avg", {"TIME": f"2030{month}01"})
    
    month_name = datetime.datetime.strptime(month, "%m").strftime("%B")
    return f"Predicted average temperature in {month_name} in year 2030: {np.array(image).mean():.2f} degrees Celsius."

@tool(parse_docstring=True)
def get_elevation_data(
    coords: Annotated[list[float], InjectedState("coords")]
) -> str:
    """Return processed data from digital elevation model. The bounding box coordinates will be provided during runtime.
    
    Args:
        coords: Map bounding box coordinates.
    """
    image = get_map(coords, "DEM_color")
    elevations = rgb2elevation(image.convert('RGB'))
    zones_data = count_elevation_zones(elevations)
    
    n_pixels = len(image.getdata())
    bbox_area = get_area(coords)
    zones_areas = {k: v / n_pixels * bbox_area for k, v in zones_data.items()}
    
    return f"Elevation data:\n"\
        + "\n".join([f"{k}: {v:.2f} km squared" for k, v in zones_areas.items() if v != 0])

@tool(parse_docstring=True)
def get_transport_data(
    coords: Annotated[list[float], InjectedState("coords")]
) -> str:
    """Return processed roads and transportation data. The bounding box coordinates will be provided during runtime.
    
    Args:
        coords: Map bounding box coordinates.
    """
    image = get_map(coords, "transport")
    roads_pixel_count = np.sum((np.array(image)[:,:,-1] > 0))
    
    return f"Area covered by roads: {roads_pixel_count / len(image.getdata()) * 100:.2f}%"

@tool(parse_docstring=True)
def get_eurostat_population_data(
    coords: Annotated[list[float], InjectedState("coords")]
) -> str:
    """Return processed eurostat data about total or employed population. The bounding box coordinates will be provided during runtime.
    
    Args:
        coords: Map bounding box coordinates.
    """
    image = get_map(coords, "EUROSTAT_2021", {"layer": "total_population_eurostat_griddata_2021"})
    total_population = np.sum(np.unique(np.array(image)))
    
    image = get_map(coords, "EUROSTAT_2021", {"layer": "employed_population_eurostat_griddata_2021"})
    employed_population = np.sum(np.unique(np.array(image)))
    
    return f"Eurostat population data:\n"\
        + f"Total population: {total_population}\n"\
        + f"Employed population: {employed_population}"

@tool(parse_docstring=True)
def estimate_hotel_suitability(
    highlighted_square: Annotated[list[float], InjectedState("highlighted_square")]
) -> str:
    """Using data about hotels and other establishments, estimate the number of hotels that could be suitable
    for a highlighted area. The highlighted square data will be provided during runtime.
    
    Args:
        highlighted_square: Highlighted square coordinates.
    """
    features = hotels_model.load_features()
    model = hotels_model.load_model()

    idx = '_'.join([str(v) for v in highlighted_square])
    square_features = features.loc[idx]

    return f"Estimated number of hotels suitable for highlighted square: {model.predict(square_features):.2f}"

def get_all_tools():
    return [
        get_open_land_use,
        get_monthly_average_temperature,
        get_monthly_average_temperature_prediction,
        get_elevation_data,
        get_transport_data,
        get_eurostat_population_data,
        estimate_hotel_suitability
    ]