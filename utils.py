
def get_api_coords(coords):
    """
    Parse the rectangle coordinates from the Folium map data into a format that can be used by the tools API.
    """
    lon_min, lat_min = coords[0]
    lon_max, lat_max = coords[2]
    return [lat_min, lon_min, lat_max, lon_max]

def get_center(coords):
    """
    Calculate the center of the rectangle based on its coordinates.
    """
    lon_min, lat_min = coords[0]
    lon_max, lat_max = coords[2]
    return [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]
