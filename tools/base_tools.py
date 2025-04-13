from typing import ClassVar, Optional

from langchain_core.tools import BaseTool
from shapely.geometry import Polygon

class GeospatialTool(BaseTool):
    """Base class for tools with geospatial boundaries."""
    boundary: ClassVar[Optional[Polygon]] = None
    
    def is_applicable(self, geom: Polygon) -> bool:
        boundary = getattr(self, "boundary", None)
        if boundary is None:
            return True
        return self.boundary.intersects(geom)