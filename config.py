import datetime
import numpy as np

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
    'DEM_MASL':
        {'wms_root_url':'https://gis.lesprojekt.cz/cgi-bin/mapserv', 
        'data':{'map':'/home/dima/maps/foodie/dem.map', 'service':'WMS', 'version':'1.3.0', 'request':'GetMap', 'bbox':'49.3,12.7,49.4,12.8', 'crs':'EPSG:4326', 'width':'1562', 'height':'680', 'layers':'DEM_ORIG', 'styles':'', 'format':'gtiff' }, 
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