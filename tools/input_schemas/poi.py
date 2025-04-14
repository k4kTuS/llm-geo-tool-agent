from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
from typing_extensions import Annotated
from typing import Optional

from schemas.geometry import BoundingBox

class PoiInput(BaseModel):
    bounding_box: Annotated[BoundingBox, InjectedState("bounding_box")] = Field(
        ...,
        description="Map bounding box coordinates."
    )
    categories: Optional[list[str]] = Field(
        default=None,
        description="Optional list of OSM keys for points of interest to search for."
    )