from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from schemas.geometry import PointMarker

class HotelSuitabilitySchema(BaseModel):
    hotel_site_marker: Annotated[PointMarker, InjectedState("hotel_site_marker")] = Field(..., description="Coordinates of a potential hotel site marker.")