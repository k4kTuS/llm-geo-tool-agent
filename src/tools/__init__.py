from .eurostat import EurostatPopulationTool
from .hotel_suitability import HotelSuitabilityTool
from .land import LandCoverTool, LandUseTool, ElevationTool
from .openmeteo import CurrentWeatherTool, WeatherForecastTool
from .poi import PoiTool, OSMTool
from .temperature import TemperatureAnalysisTool, TemperatureLongPredictionTool
from .tourism import TourismTool


def get_all_tools():
    return [
        EurostatPopulationTool(),
        HotelSuitabilityTool(),
        LandCoverTool(),
        LandUseTool(),
        ElevationTool(),
        # PoiTool(),
        OSMTool(),
        TemperatureAnalysisTool(),
        TemperatureLongPredictionTool(),
        CurrentWeatherTool(),
        WeatherForecastTool(),
        TourismTool(),
    ]