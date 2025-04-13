from agents.tool_filters.base import BaseToolFilter
from schemas.geometry import BoundingBox

class GeospatialToolFilter(BaseToolFilter):
    def filter(self, tools, context):
        bounding_box: BoundingBox = context.get("bounding_box")
        if not bounding_box:
            return tools

        filtered = []
        for tool in tools:
            if tool.is_applicable(bounding_box.geom):
                filtered.append(tool)
            else:
                print(f"Tool {tool.name} is not applicable for the geometry")
        return filtered