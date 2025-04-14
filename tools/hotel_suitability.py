from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel
from shapely.geometry import box

from models import hotels_model
from tools.base import GeospatialTool
from tools.input_schemas.hotel import HotelSuitabilitySchema
from schemas.geometry import PointMarker
from utils.tool import find_square_for_marker


class HotelSuitabilityTool(GeospatialTool):
    name: str = "estimate_hotel_suitability"
    description: str = "Using data about hotels and other establishments, estimate the number of hotels that could be suitable for the marked site provided during runtime."
    args_schema: Optional[Type[BaseModel]] = HotelSuitabilitySchema
    boundary = box(12.09,48.55,18.87,51.06)

    def _run(self, hotel_site_marker: PointMarker):
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