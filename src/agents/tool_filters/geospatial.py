from agents.tool_filters.base import BaseToolFilter
from schemas.geometry import BoundingBox

class GeospatialToolFilter(BaseToolFilter):
    def filter(self, tools, context):
        bounding_box: BoundingBox = context.get("bounding_box")
        if not bounding_box:
            return tools

        filtered = []

        if bounding_box.crs != "EPSG:4326":
            bounding_box = bounding_box.to_crs("EPSG:4326")
        geom = bounding_box.geom

        for tool in tools:
            if tool.is_applicable(geom):
                filtered.append(tool)
            else:
                print(f"Tool {tool.name} is not applicable for the geometry")
        return filtered