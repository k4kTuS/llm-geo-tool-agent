from .eurostat_tool import EurostatPopulationTool
from .hotel_suitability_tool import HotelSuitabilityTool
from .land_tools import LandCoverTool, LandUseTool, ElevationTool
from .openmeteo_tool import CurrentWeatherTool, WeatherForecastTool
from .spoi_tool import SpoiTool
from .temperature_tools import TemperatureAnalysisTool, TemperatureLongPredictionTool
from .tourism_tool import TourismTool


def get_all_tools():
    return [
        EurostatPopulationTool(),
        HotelSuitabilityTool(),
        LandCoverTool(),
        LandUseTool(),
        ElevationTool(),
        SpoiTool(),
        TemperatureAnalysisTool(),
        TemperatureLongPredictionTool(),
        CurrentWeatherTool(),
        WeatherForecastTool(),
        TourismTool(),
    ]