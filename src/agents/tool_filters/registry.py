from agents.tool_filters import (
    BaseToolFilter,
    GeospatialToolFilter,
    SemanticToolFilter
)

FILTER_REGISTRY: dict[str, BaseToolFilter] = {
    "geospatial": GeospatialToolFilter,
    "semantic": SemanticToolFilter,
}