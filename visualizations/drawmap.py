import folium
from folium.plugins import Draw
from folium.utilities import JsCode

from project_paths import ASSETS_DIR

class DrawMap:
    def __init__(self, location=(49.75, 13.39), zoom_start=13):
        self.map_ = folium.Map(location=location, zoom_start=zoom_start)
        self.add_draw()
    
    def add_draw(self):
        draw = Draw(
            draw_options={
                'polyline': False,
                'polygon': False,
                'circle': False,
                'circlemarker': False,
                'marker': True, # Potentional hotel site
                'rectangle': True, # Area of interest
            },
            edit_options={
                'edit': True,
                'remove': True,
            },
            # Custom JS that ensures only one rectangle is present at a time on the map
            on={
                "add": JsCode(open(f"{ASSETS_DIR}/js/handleAddDrawing.js", "r").read()),
                "remove": JsCode(open(f"{ASSETS_DIR}/js/handleRemoveDrawing.js", "r").read()),
            }
        )
        draw.add_to(self.map_)
