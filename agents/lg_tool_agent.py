import configparser

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from typing_extensions import Annotated, TypedDict, Optional
import streamlit as st

from paths import PROJECT_ROOT
from tools import get_all_tools
from utils.geometry_utils import BoundingBox, PointMarker

cfg = configparser.ConfigParser()
cfg.read(f'{PROJECT_ROOT}/config.ini')

SYSTEM_MESSAGE_TEMPLATE = """
You are a helpful assistant working with geographical data. Some questions will be tied to an area defined by bounding box coordinates.
These coordinates represent a geographical area on the map. You do not need to ask for the coordinates; allways assume you already know the coordinates of the area you are working with.

Whenever the task involves retrieving information about a location, assume the location is within the area defined by the bounding box coordinates.
Do not copy the data obtained from the tools directly to the response, but rather summarize it in a way that is easy to understand for the user.
"""

class AgentState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: List of current chat messages
        bounding_box: Bounding box instance representing a geographical area of interest.
        hotel_site_marker: Coordinates of a potential hotel site marker.
    """
    messages: Annotated[list, add_messages]
    bounding_box: BoundingBox
    hotel_site_marker: PointMarker

def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if f'chat_history_{session_id}' not in st.session_state:
        st.session_state[f'chat_history_{session_id}'] = InMemoryChatMessageHistory()
    return st.session_state[f'chat_history_{session_id}']

def clear_chat_history(session_id: str):
    st.session_state[f'chat_history_{session_id}'] = InMemoryChatMessageHistory()

@st.cache_resource
def build_graph():
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
            return "tools"

        get_chat_history(config["configurable"]["session_id"]).add_messages(msgs)
        return END

    def call_model(state: AgentState, config: RunnableConfig):
        chat_history = get_chat_history(config["configurable"]["session_id"])
        msgs = [SystemMessage(content=SYSTEM_MESSAGE_TEMPLATE)] + list(chat_history.messages) + state["messages"]
        response = llm_with_tools.with_config({"run_name": cfg['LANGSMITH']['model_run_name']}).invoke(msgs)
        return {"messages": [response]}

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(get_all_tools()))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")

    app = workflow.compile()
    return app