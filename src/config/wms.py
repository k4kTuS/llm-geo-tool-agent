import datetime
import numpy as np
from schemas.remapping import RangeMapping, RangeMappingSet

wms_config={ 
    'climate_era5_temperature_last_5yrs_month_avg':
        {'wms_root_url':'https://olu.lesprojekt.cz/cgi-bin/mapserv', 
        'data':{'map':'/data/maps/thematic_maps.map', 'service':'WMS', 'version':'1.3.0', 'request':'GetMap', 'bbox':'49.3,12.7,49.4,12.8', 'crs':'EPSG:4326', 'width':'1562', 'height':'680', 'layers':'t2m_2020', 'TIME':'20200101','styles':'', 'format':'gtiff' }, 
        'alternatives':{'TIME':[datetime.date(2020,i,1).strftime('%Y%m%d') for i in range(1,13)]}
        }, 
    'climate_ipcc_rcp45_temperature_2030s_month_avg':
        {'wms_root_url':'https://olu.lesprojekt.cz/cgi-bin/mapserv', 
        'data':{'map':'/data/maps/thematic_maps.map', 'service':'WMS', 'version':'1.3.0', 'request':'GetMap', 'bbox':'49.3,12.7,49.4,12.8', 'crs':'EPSG:4326', 'width':'1562', 'height':'680', 'layers':'tas_2030', 'TIME':'20300101','styles':'', 'format':'gtiff' }, 
        'alternatives':{'TIME':[datetime.date(2030,i,1).strftime('%Y%m%d') for i in range(1,13)]}
        }, 
    'OLU_EU':
        {'wms_root_url':'https://olu.lesprojekt.cz/cgi-bin/mapserv', 
        'data':{'map':'/data/maps/olu_europe.map', 'service':'WMS', 'version':'1.3.0', 'request':'GetMap', 'bbox':'49.3,12.7,49.4,12.8', 'crs':'EPSG:4326', 'width':'1562', 'height':'680', 'layers':'olu_obj_lu', 'styles':'', 'format':'png' }, 
        'alternatives':{'layers':['olu_obj_lu', 'olu_obj_lc']}
        },
    'EUROSTAT_2021':
        {'wms_root_url':'https://olu.lesprojekt.cz/cgi-bin/mapserv', 
        'data':{'map':'/data/maps/thematic_maps.map', 'service':'WMS', 'version':'1.3.0', 'request':'GetMap', 'bbox':'49.3,12.7,49.4,12.8', 'crs':'EPSG:4326', 'width':'1562', 'height':'680', 'layers':'total_population_eurostat_griddata_2021', 'styles':'', 'format':'gtiff' }, 
        'alternatives':{'layers':['total_population_eurostat_griddata_2021', 'employed_population_eurostat_griddata_2021']}
        }, 
    'DEM_MASL':
        {'wms_root_url':'https://gis.lesprojekt.cz/cgi-bin/mapserv', 
        'data':{'map':'/home/dima/maps/foodie/dem.map', 'service':'WMS', 'version':'1.3.0', 'request':'GetMap', 'bbox':'49.3,12.7,49.4,12.8', 'crs':'EPSG:4326', 'width':'1562', 'height':'680', 'layers':'DEM_ORIG', 'styles':'', 'format':'gtiff' }, 
        'alternatives':{}
        },
}

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
    'Not Known Use': (240,240,240),
    'Missing Data': (255,255,255),
}
rgb_LU_mapping = {v: k for k, v in LU_rgb_mapping.items()}

LC_rgb_mapping = {
    'Water bodies': (128, 242, 230),
    'Pastures': (230, 230, 77),
    'Continuous urban fabric': (230, 0, 77),
    'Discontinuous urban fabric': (255, 0, 0),
    'Industrial or commercial units': (204, 77, 242),
    'Road and rail networks and associated land': (204, 0, 0),
    'Port areas': (230, 204, 204),
    'Airports': (230, 204, 230),
    'Mineral extraction sites': (166, 0, 204),
    'Dump sites': (166, 77, 0),
    'Construction sites': (255, 77, 255),
    'Green urban areas': (255, 166, 255),
    'Sport and leisure facilities': (255, 230, 255),
    'Non-irrigated arable land': (255, 255, 168),
    'Permanently irrigated land': (255, 255, 0),
    'Rice fields': (230, 230, 0),
    'Vineyards': (230, 128, 0),
    'Fruit trees and berry plantations': (242, 166, 77),
    'Olive groves': (230, 166, 0),
    'Annual crops associated with permanent crops': (255, 230, 166),
    'Complex cultivation patterns': (255, 230, 77),
    'Land principally occupied by agriculture, with significant areas of natural vegetation': (230, 204, 77),
    'Agro-forestry areas': (242, 204, 166),
    'Broad-leaved forest': (128, 255, 0),
    'Coniferous forest': (0, 166, 0),
    'Mixed forest': (77, 255, 0),
    'Natural grasslands': (204, 242, 77),
    'Moors and heathland': (166, 255, 128),
    'Sclerophyllous vegetation': (166, 230, 77),
    'Transitional woodland-shrub': (166, 242, 0),
    'Beaches, dunes, sands': (230, 230, 230),
    'Bare rocks': (204, 204, 204),
    'Sparsely vegetated areas': (204, 255, 204),
    'Burnt areas': (0, 0, 0),
    'Glaciers and perpetual snow': (166, 230, 204),
    'Inland marshes': (166, 166, 255),
    'Peat bogs': (77, 77, 255),
    'Salt marshes': (204, 204, 255),
    'Salines': (230, 230, 255),
    'Intertidal flats': (166, 166, 230),
    'Water courses': (0, 204, 242),
    'Coastal lagoons': (0, 255, 166),
    'Estuaries': (166, 255, 230),
    'Sea and ocean': (230, 242, 255),
    'UNCLASSIFIED LAND SURFACE': (200, 200, 200),
    'UNCLASSIFIED WATER BODIES': (230, 242, 255),
    'Missing Data': (255,255,255),
 }
rgb_LC_mapping = {v: k for k, v in LC_rgb_mapping.items()}

elevation_range_set = RangeMappingSet(
    name="elevation",
    ranges=[
        RangeMapping(min=-500, max=0, label="Below Sea Level"),
        RangeMapping(min=0, max=200, label="Lowlands"),
        RangeMapping(min=200, max=500, label="Hills & Low Plateaus"),
        RangeMapping(min=500, max=1000, label="High Hills & Plateaus"),
        RangeMapping(min=1000, max=2000, label="Low Mountains"),
        RangeMapping(min=2000, max=3000, label="Mid Mountains"),
        RangeMapping(min=3000, max=4500, label="High Mountains"),
        RangeMapping(min=4500, max=6000, label="Alpine/Glacial Zone"),
        RangeMapping(min=6000, max=np.inf, label="Extreme High Peaks"),
    ],
    unit="meters"
)