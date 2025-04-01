from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal

from schemas.geometry import BoundingBox


class OpenmeteoForecastInput(BaseModel):
    bounding_box: Annotated[BoundingBox, InjectedState("bounding_box")] = Field(..., description="Map bounding box coordinates.")
    date_start: str = Field(
        ...,
        description=("The start date for the forecast in YYYY-MM-DD format.")
    )
    date_end: str = Field(
        ...,
        description=("The end date for the forecast in YYYY-MM-DD format.")
    )
    forecast_type: Literal["hourly", "daily"] = Field(
        ...,
        description=(
             "Type of forecast data to retrieve. "
            "'hourly' provides detailed weather data for each hour, ideal for short-term planning and real-time analysis. "
            "'daily' offers aggregated weather summaries for each day, useful for long-term forecasts and general weather trends."
        )
    )