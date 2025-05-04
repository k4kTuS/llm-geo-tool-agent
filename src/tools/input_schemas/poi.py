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
        description="Optional list of category keys for points of interest to search for."
    )
    relevance_threshold: Optional[float] = Field(
        description=(
        "Similarity threshold between input categories and categories of retrieved points of interest, ranging from 0.7 to 1.0. "
        "Use it to filter out less relevant categories."
        )
    )

class OSMInput(BaseModel):
    bounding_box: Annotated[BoundingBox, InjectedState("bounding_box")] = Field(
        ...,
        description="Map bounding box coordinates."
    )
    query: str = Field(
        ...,
        description="A complete natural language sentence describing what to find."
    )