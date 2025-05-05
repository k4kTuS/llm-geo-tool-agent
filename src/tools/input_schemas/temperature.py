from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from schemas.geometry import BoundingBox


class TemperatureAnalysisInput(BaseModel):
    bounding_box: Annotated[BoundingBox, InjectedState("bounding_box")] = Field(..., description="Map bounding box coordinates.")
    month: int = Field(..., description="Month as an integer (1-12).")