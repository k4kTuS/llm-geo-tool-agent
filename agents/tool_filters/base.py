from abc import ABC, abstractmethod
from typing import List, Any

from tools.base_tools import GeospatialTool

class BaseToolFilter(ABC):
    @abstractmethod
    def filter(self, tools: List[GeospatialTool], context: dict[str, Any]) -> List[GeospatialTool]:
        pass