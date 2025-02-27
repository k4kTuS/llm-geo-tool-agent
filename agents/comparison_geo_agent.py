import configparser

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from typing_extensions import Annotated, TypedDict, Optional
import streamlit as st

from paths import PROJECT_ROOT
from tools import get_all_tools
from schemas.geometry import BoundingBox, PointMarker
from utils.agent_utils import get_chat_history

cfg = configparser.ConfigParser()
cfg.read(f'{PROJECT_ROOT}/config.ini')

SYSTEM_MESSAGE = """
You are a helpful assistant working with geographical data. Some questions will be tied to an area defined by bounding box coordinates.
These coordinates represent a geographical area on the map. You do not need to ask for the coordinates; allways assume you already know the coordinates of the area you are working with.

Whenever the task involves retrieving information about a location, assume the location is within the area defined by the bounding box coordinates.
If you use tools, do not copy the data obtained from the tools directly to the response, but rather summarize it in a way that is easy to understand for the user.
Include your own information and insights in the response, do not rely solely on tools.
"""

class AgentState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: List of current chat messages
        bounding_box: Bounding box instance representing a geographical area of interest.
        hotel_site_marker: Coordinates of a potential hotel site marker.
        alternative_response: Alternative model response generated without tools
    """
    messages: Annotated[list[AnyMessage], add_messages]
    bounding_box: BoundingBox
    hotel_site_marker: PointMarker
    alternative_response: Optional[AIMessage]

llm = ChatOpenAI(
    model=cfg['OPENAI']['model_id'],
    temperature=0,
    max_retries=2,
)

llm_with_tools = llm.bind_tools(get_all_tools())

def should_continue(state: AgentState, config: RunnableConfig):
    msgs = state["messages"]
    last_message = msgs[-1]
    if last_message.tool_calls:
        if state.get("alternative_response", None) is None:
            return ["alternative", "tools"]
        return "tools"

    get_chat_history().add_messages(msgs)
    last_message.run_id = config["configurable"]["run_id"]
    for m in msgs:
        st.session_state["all_messages"][m.id] = m
    alternative = state.get("alternative_response", None)
    if  alternative is not None:
        last_message.alternative_id = alternative.id
        alternative.alternative_id = last_message.id
        alternative.run_id = config["configurable"]["run_id"]
        st.session_state["all_messages"][alternative.id] = alternative
    return END

def call_model(state: AgentState, config: RunnableConfig):
    chat_history = get_chat_history()
    msgs = [SystemMessage(content=SYSTEM_MESSAGE)] + list(chat_history.messages) + state["messages"]
    response = llm_with_tools.invoke(msgs)
    return {"messages": [response]}

def call_without_tools(state: AgentState, config: RunnableConfig):
    bbox_text = f"The bounding box is defined by the following coordinates (lat1, lon1, lat2, lon2):\n" \
                f"{state['bounding_box'].to_string_latlon()}\n"
    user_msg = HumanMessage(content=bbox_text + state["messages"][0].content)

    msgs = [user_msg]
    response = llm.invoke(msgs)
    return {"alternative_response": response}

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(get_all_tools()))
workflow.add_node("alternative", call_without_tools)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", "alternative", END])
workflow.add_edge("tools", "agent")

comparison_geo_agent = workflow.compile()