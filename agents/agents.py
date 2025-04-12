from agents import comparison_geo_agent
from agents import geo_agent
from agents import agent_tool_selector

def get_agent(name: str):
    if name == "geo":
        return geo_agent.graph
    elif name == "comparison_geo":
        return comparison_geo_agent.graph
    elif name == "tool_selector_geo":
        return agent_tool_selector.graph
    else:
        raise ValueError(f"Unknown agent name: {name}")