import streamlit as st

from langchain_core.messages import BaseMessage

from agent import get_chat_history

def get_api_coords(coords):
    """
    Parse the rectangle coordinates from the Folium map data into a format that can be used by the tools API.
    """
    lon_min, lat_min = coords[0]
    lon_max, lat_max = coords[2]
    return [lat_min, lon_min, lat_max, lon_max]

def get_center(coords):
    """
    Calculate the center of the rectangle based on its coordinates.
    """
    lon_min, lat_min = coords[0]
    lon_max, lat_max = coords[2]
    return [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]

def parse_drawing_coords(map_data, drawing_type):
    if not map_data["all_drawings"]:
        return None
    count = 0
    last_drawing = None
    for drawing in map_data["all_drawings"]:
        if drawing["geometry"]["type"] == drawing_type:
            count = count + 1
            last_drawing = drawing
    if count != 1:
        return None
    return last_drawing["geometry"]["coordinates"]

def write_message(message: BaseMessage):
    if message.type == "human":
        st.chat_message("human").write(message.content)
        return

    if message.type == "ai" and message.content != "":
        st.chat_message("ai", avatar="🌿").markdown(message.content.replace("\n", "  \n"), unsafe_allow_html=True)

    has_tool_info = "tool_calls" in message.additional_kwargs or message.type == "tool"
    if has_tool_info and st.session_state["show_tool_calls"]:
        st.chat_message("tool", avatar="🛠️").markdown(message.pretty_repr().replace("\n", "  \n"), unsafe_allow_html=True)

def rewrite_chat_history():
    chat_history = get_chat_history(st.session_state["user"])
    for m in chat_history.messages:
        write_message(m)