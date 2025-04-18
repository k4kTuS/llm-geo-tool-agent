from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated, TypedDict, Optional

from agents.prompts import SYSTEM_PROMP_GEO, get_current_timestamp
from tools import get_all_tools
from schemas.geometry import BoundingBox, PointMarker
from utils.agent import get_llm

class AgentState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: List of current chat messages
        chat_history: List of chat history messages
        bounding_box: Bounding box instance representing a geographical area of interest.
        point_marker: Point marker representing a specific location in addition to the bounding box.
    """
    messages: Annotated[list[AnyMessage], add_messages]
    chat_history: list[AnyMessage]
    bounding_box: BoundingBox
    point_marker: PointMarker

def should_continue(state: AgentState, config: RunnableConfig):
    msgs = state["messages"]
    last_message = msgs[-1]
    if last_message.tool_calls:
        return "tools"
    return END

def call_model(state: AgentState, config: RunnableConfig):
    llm = get_llm(config["configurable"]["model_name"])
    llm_with_tools = llm.bind_tools(get_all_tools())

    msgs = [SystemMessage(content=SYSTEM_PROMP_GEO.format(timestamp=get_current_timestamp()))] + state["chat_history"] + state["messages"]
    response = llm_with_tools.invoke(msgs)
    return {"messages": [response]}

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(get_all_tools()))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

graph = workflow.compile()