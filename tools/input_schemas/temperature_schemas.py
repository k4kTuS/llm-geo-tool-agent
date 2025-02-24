from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from schemas.geometry import BoundingBox


class TemperatureAnalysisInput(BaseModel):
    bounding_box: Annotated[BoundingBox, InjectedState("bounding_box")] = Field(..., description="Map bounding box coordinates.")
    month: str = Field(..., description="Month in the format of MM.")


class TemeperatureForecastInput(BaseModel):
    bounding_box: Annotated[BoundingBox, InjectedState("bounding_box")] = Field(..., description="Map bounding box coordinates.")
    forecast_days: int = Field(..., description="Number of days to forecast. Use 0 for current temperature. The maximum is 16 days.")