from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated, TypedDict, Optional

from agents.prompts import SYSTEM_PROMPT_COMPARISON, get_current_timestamp
from agents.tool_filters import FILTER_REGISTRY, BaseToolFilter
from tools import get_all_tools
from schemas.geometry import BoundingBox, PointMarker
from utils.agent import get_llm

_all_tools = get_all_tools()

class AgentState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: List of current chat messages
        chat_history: List of chat history messages
        bounding_box: Bounding box instance representing a geographical area of interest.
        point_marker: Point marker representing a specific location in addition to the bounding box.
        alternative_user_message: User message for the alternative response
        alternative_response: Alternative model response generated without tools
        alternative_history: Message history for the alternative response
        selected_tools: List of tools that were selected by the tool selector
    """
    messages: Annotated[list[AnyMessage], add_messages]
    chat_history: list[AnyMessage]
    bounding_box: BoundingBox
    point_marker: PointMarker
    alternative_user_message: Optional[str]
    alternative_response: Optional[AIMessage]
    alternative_history: Optional[list[AnyMessage]]
    selected_tools: list[str]

def select_tools(state: AgentState, config: RunnableConfig):
    filter_names = config["configurable"].get("filters", ["semantic"])

    tools = _all_tools
    for filter_name in filter_names:
        filter: BaseToolFilter = FILTER_REGISTRY[filter_name]()
        tools = filter.filter(tools, state)

    return {"selected_tools": [tool.name for tool in tools]}

def should_continue(state: AgentState, config: RunnableConfig):
    msgs = state["messages"]
    last_message = msgs[-1]
    if last_message.tool_calls:
        if state.get("alternative_response", None) is None:
            return ["alternative", "tools"]
        return "tools"
    return END

def call_model(state: AgentState, config: RunnableConfig):
    llm = get_llm(config["configurable"]["model_name"])

    msgs = [SystemMessage(content=SYSTEM_PROMPT_COMPARISON.format(timestamp=get_current_timestamp()))] + state["chat_history"] + state["messages"]

    selected_tools = state["selected_tools"]
    print("Generating with selected tools:", selected_tools)
    llm_with_tools = llm.bind_tools([tool for tool in _all_tools if tool.name in selected_tools])

    response = llm_with_tools.invoke(msgs)
    return {"messages": [response]}

def call_without_tools(state: AgentState, config: RunnableConfig):
    llm = get_llm(config["configurable"]["model_name"])

    msgs = state["alternative_history"] + [state["alternative_user_message"]]
    response = llm.invoke(msgs)
    return {"alternative_response": response}

workflow = StateGraph(AgentState)

workflow.add_node("tool_selection", select_tools)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(_all_tools))
workflow.add_node("alternative", call_without_tools)

workflow.add_edge(START, "tool_selection")
workflow.add_edge("tool_selection", "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", "alternative", END])
workflow.add_edge("tools", "agent")

graph = workflow.compile()