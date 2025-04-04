from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated, TypedDict, Optional
import streamlit as st

from agents.prompts import SYSTEM_PROMPT_COMPARISON, get_current_timestamp
from agents.tool_selector import filter_tools
from paths import PROJECT_ROOT
from tools import get_all_tools
from schemas.geometry import BoundingBox, PointMarker
from utils.agent_utils import get_chat_history, get_llm

class AgentState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: List of current chat messages
        bounding_box: Bounding box instance representing a geographical area of interest.
        hotel_site_marker: Coordinates of a potential hotel site marker.
        alternative_response: Alternative model response generated without tools
        selected_tools: List of tools that were selected by the tool selector
    """
    messages: Annotated[list[AnyMessage], add_messages]
    bounding_box: BoundingBox
    hotel_site_marker: PointMarker
    alternative_response: Optional[AIMessage]
    selected_tools: list[str]

def select_tools(state: AgentState):
    last_user_message = state["messages"][-1]
    query = last_user_message.content
    selected_tools = filter_tools([tool.name for tool in get_all_tools()], query)
    return {"selected_tools": selected_tools}

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
        st.session_state.all_messages[m.id] = m
    alternative = state.get("alternative_response", None)
    if  alternative is not None:
        last_message.alternative_id = alternative.id
        alternative.alternative_id = last_message.id
        alternative.run_id = config["configurable"]["run_id"]
        st.session_state.all_messages[alternative.id] = alternative
    return END

def call_model(state: AgentState, config: RunnableConfig):
    llm = get_llm(config["configurable"]["model_name"])

    chat_history = get_chat_history()
    msgs = [SystemMessage(content=SYSTEM_PROMPT_COMPARISON.format(timestamp=get_current_timestamp()))] + list(chat_history.messages) + state["messages"]

    selected_tools = state["selected_tools"]
    print("Generating with selected tools:", selected_tools)
    llm_with_tools = llm.bind_tools([tool for tool in get_all_tools() if tool.name in selected_tools])

    response = llm_with_tools.invoke(msgs)
    return {"messages": [response]}

def call_without_tools(state: AgentState, config: RunnableConfig):
    llm = get_llm(config["configurable"]["model_name"])

    bbox_text = f"The bounding box is defined by the following coordinates (lat1, lon1, lat2, lon2):\n" \
                f"{state['bounding_box'].to_string_latlon()}\n"
    user_msg = HumanMessage(content=bbox_text + state["messages"][0].content)

    msgs = [user_msg]
    response = llm.invoke(msgs)
    return {"alternative_response": response}

workflow = StateGraph(AgentState)

workflow.add_node("tool_selection", select_tools)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(get_all_tools()))
workflow.add_node("alternative", call_without_tools)

workflow.add_edge(START, "tool_selection")
workflow.add_edge("tool_selection", "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", "alternative", END])
workflow.add_edge("tools", "agent")

agent_tool_selector = workflow.compile()