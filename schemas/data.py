from dataclasses import dataclass
from typing import Literal, Any

@dataclass
class DataResponse:
    name: str
    source: str
    data_type: Literal["text", "image", "dataframe", "table"]
    data: Any
    show_data: bool