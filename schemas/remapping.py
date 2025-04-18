from pydantic import BaseModel, Field
import numpy as np

class RangeMapping(BaseModel):
    min: float = Field(..., description="Lower bound of the range (inclusive)")
    max: float = Field(..., description="Upper bound of the range (exclusive)")
    label: str = Field(..., description="Label assigned to the range")

    def contains(self, value: float) -> bool:
        return self.min <= value < self.max

class RangeMappingSet(BaseModel):
    name: str = Field(..., description="Name of the mapping set")
    ranges: list[RangeMapping] = Field(..., description="List of range mappings")
    unit: str = Field(..., description="Unit of measurement for the ranges")

    def get_label(self, value: float) -> str | None:
        for r in self.ranges:
            if r.contains(value):
                return r.label
        return None
