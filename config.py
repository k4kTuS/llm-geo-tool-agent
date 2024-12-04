import datetime

import numpy as np

map_config={ 
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

wfs_config={
    'SPOI':
        {'wfs_root_url':'https://gis.lesprojekt.cz/cgi-bin/mapserv',
        'data':{'map':'/home/dima/maps/spoiky.map', 'service':'WFS', 'version':'1.1.0', 'request':'GetFeature', 'srsname':'EPSG:4326', 'typename':'spoi', 'outputformat':'geojson'}
        }
}

# Used with DEM
elevation_ranges = [
    (0, 50, "Coastal Plain"),
    (50, 100, "Floodplain"),
    (100, 200, "Low Coastal Plateau"),
    (200, 300, "Interior Plateau"),
    (300, 500, "Low Hill Country"),
    (500, 700, "High Hill Country"),
    (700, 900, "Low Mountainous Area"),
    (900, 1200, "Moderate Mountainous Area"),
    (1200, 1500, "High Mountainous Area"),
    (1500, 1800, "Subalpine Low"),
    (1800, 2100, "Subalpine High"),
    (2100, 2400, "Lower Alpine Zone"),
    (2400, 2700, "Mid Alpine Zone"),
    (2700, 3000, "Upper Alpine Zone"),
    (3000, 3300, "Lower Subnival Zone"),
    (3300, 3600, "Mid Subnival Zone"),
    (3600, 4000, "Upper Subnival Zone"),
    (4000, 4500, "Lower Nival Zone"),
    (4500, 5000, "Upper Nival Zone"),
    (5000, 5500, "Lower Glacial Zone"),
    (5500, 6000, "Mid Glacial Zone"),
    (6000, np.inf, "Upper Glacial Zone")
]

# Used with OLU
LU_rgb_mapping = {
    'Primary Production': (180,230,110),
    'Agriculture': (230,230,110),
    'Forestry': (110,230,110),
    'Mining And Quarrying': (180,200,110),
    'Secondary Production': (100,100,100),
    'Manufacturing Of Wood And Wood Based Products': (140,100,100),
    'Manufacturing Of Machinery': (50,100,100),
    'Energy Production': (150,150,40),
    'Other Industry': (100,100,100),
    'Tertiary Production': (150,150,150),
    'Commercial Services': (150,170,130),
    'Financial Professional And Information Services': (190,170,130),
    'Community Services': (210,170,150),
    'Cultural Entertainment And Recreational Services': (120,170,150),
    'Transport Network Logistics And Utilities': (180,120,240),
    'Transport Networks': (220,160,220),
    'Water Transport': (140,120,240),
    'Logistical And Storage Services': (230,200,200),
    'Utilities': (250,220,220),
    'Residential Use': (240,120,100),
    'Permanent Residential Use': (240,60,40),
    'Residential Use With Other Compatible Uses': (240,84,100),
    'Other Uses': (220,220,220),
    'Transitional Areas / Natural Areas Not In Other Economic Use': (200,220,220),
    'Land Areas Not In Other Economic Use': (200,255,200),
    'Water Areas Not In Other Economic Use': (200,200,255),
    'Areas Without Any Specified Planned Land Use': (230,240,240),
    'Not Known Use': (240,240,240)
}
rgb_LU_mapping = {v: k for k, v in LU_rgb_mapping.items()}