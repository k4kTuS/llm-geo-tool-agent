from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
import streamlit as st

from agents.graphs import (
    comparison_geo_graph,
    filtering_graph,
    geo_graph
)
from schemas.geometry import BoundingBox, PointMarker
from utils.agent import get_chat_history, pair_response_messages, store_run_messages

class BaseAgent(ABC):
    def __init__(self, name: str, graph: CompiledStateGraph):
        self.name = name
        self.graph = graph
    
    @abstractmethod
    def run(self, prompt: str, bounding_box: BoundingBox, marker: PointMarker | None, config: dict):
        pass
    
    @abstractmethod
    def store_run_details(self, state: dict, config: dict):
        pass

class GeoAgent(BaseAgent):
    def run(self, prompt: str, bounding_box: BoundingBox, marker: PointMarker | None, config: dict):
        input = {
            "messages": [HumanMessage(content=prompt)],
            "chat_history": get_chat_history().messages,
            "bounding_box": bounding_box,
            "point_marker": marker
        }
        return self.graph.stream(
            input=input,
            config=config,
            stream_mode="values",
        )
    
    def store_run_details(self, state: dict, config: dict):
        get_chat_history().add_messages(state["messages"])
        last_message = state["messages"][-1]
        last_message.run_id = config["run_id"]
    
class ComparisonGeoAgent(BaseAgent):
    def run(self, prompt: str, bounding_box: BoundingBox, marker: PointMarker | None, config: dict):
        bbox_text = f"The bounding box is defined by the following coordinates (lat1, lon1, lat2, lon2):\n" \
        f"{bounding_box.to_string_yx()}\n"

        input = {
            "messages": [HumanMessage(content=prompt)],
            "chat_history": get_chat_history().messages,
            "bounding_box": bounding_box,
            "point_marker": marker,
            "alternative_user_message": HumanMessage(content=bbox_text + prompt),
            "alternative_history": get_chat_history(alternative=True).messages,
        }
        return self.graph.stream(
            input=input,
            config=config,
            stream_mode="values",
        )

    def store_run_details(self, state: dict, config: dict):
        store_run_messages(state["messages"], alternative=False)
        
        last_message = state["messages"][-1]
        last_message.run_id = config["run_id"]
        alternative = state.get("alternative_response", None)

        if alternative is not None:
            pair_response_messages(config["run_id"], last_message, alternative)
            store_run_messages([state["alternative_user_message"], state["alternative_response"]], alternative=True)

class AgentToolSelector(BaseAgent):
    def run(self, prompt: str, bounding_box: BoundingBox, marker: PointMarker | None, config: dict):
        bbox_text = f"The bounding box is defined by the following coordinates (lat1, lon1, lat2, lon2):\n" \
        f"{bounding_box.to_string_yx()}\n"

        input = {
            "messages": [HumanMessage(content=prompt)],
            "chat_history": get_chat_history().messages,
            "bounding_box": bounding_box,
            "point_marker": marker,
            "alternative_user_message": HumanMessage(content=bbox_text + prompt),
            "alternative_history": get_chat_history(alternative=True).messages,
        }
        config["configurable"]["filters"] = ["geospatial", "semantic"]
        return self.graph.stream(
            input=input,
            config=config,
            stream_mode="values",
        )
    
    def store_run_details(self, state: dict, config: dict):
        store_run_messages(state["messages"], alternative=False)
        
        last_message = state["messages"][-1]
        last_message.run_id = config["run_id"]
        alternative = state.get("alternative_response", None)

        if alternative is not None:
            pair_response_messages(config["run_id"], last_message, alternative)
            store_run_messages([state["alternative_user_message"], state["alternative_response"]], alternative=True)

def get_agent(name: str):
    if name == "geo":
        return GeoAgent(name, geo_graph.graph)
    elif name == "comparison_geo":
        return ComparisonGeoAgent(name, comparison_geo_graph.graph)
    elif name == "tool_selector_geo":
        return AgentToolSelector(name, filtering_graph.graph)
    else:
        raise ValueError(f"Unknown agent name: {name}")