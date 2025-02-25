from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal

from schemas.geometry import BoundingBox


class OpenmeteoForecastInput(BaseModel):
    bounding_box: Annotated[BoundingBox, InjectedState("bounding_box")] = Field(..., description="Map bounding box coordinates.")
    forecast_days: int = Field(..., description="Number of days to forecast. The minimum is 1 and the maximum is 16 days.")
    forecast_type: Literal["hourly", "daily"] = Field(..., description="Type of forecast data to retrieve.")