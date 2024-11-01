
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

def parse_drawing_coords(map_data, drawing_type):
    if not map_data["all_drawings"]:
        return None
    count = 0
    last_drawing = None
    for drawing in map_data["all_drawings"]:
        if drawing["geometry"]["type"] == drawing_type:
            count = count + 1
            last_drawing = drawing
    if count != 1:
        return None
    return last_drawing["geometry"]["coordinates"]
