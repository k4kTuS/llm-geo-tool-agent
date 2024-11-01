import configparser
import os
import folium
import streamlit as st

from folium.plugins import Draw
from folium.utilities import JsCode

from langchain_core.messages import HumanMessage, SystemMessage

from streamlit_folium import st_folium
from agent import build_graph, get_chat_history, clear_chat_history, SYSTEM_MESSAGE_TEMPLATE
from utils import *

CHAT_USER = "testingUser1"

config = configparser.ConfigParser()
config.read('config.ini')

os.environ['OPENAI_API_KEY'] = config['OPENAI']['api_key']

def write_message(message):
    if message.type == "human":
        st.chat_message("human").write(message.content)
        return

    has_tool_info = "tool_calls" in message.additional_kwargs or message.type == "tool"
    if has_tool_info and st.session_state["show_tool_calls"]:
        st.chat_message("tool", avatar="🛠️").markdown(message.pretty_repr().replace("\n", "  \n"), unsafe_allow_html=True)
    elif message.type == "ai" and not has_tool_info:
        st.chat_message("ai", avatar="🌿").markdown(message.content.replace("\n", "  \n"), unsafe_allow_html=True)

def rewrite_chat_history():
    chat_history = get_chat_history(CHAT_USER)
    for m in chat_history.messages:
        write_message(m)

st.title("🌿 PoliRuralPlus Chat Assistant")

st.session_state["chat_prompted"] = False

with st.sidebar as sidebar:
    st.subheader("Chat management")
    
    tools_on = st.toggle("Show tool calls details")

    if tools_on:
        st.session_state["show_tool_calls"] = True
    else:
        st.session_state["show_tool_calls"] = False

    if st.button("Clear chat history"):
        clear_chat_history(CHAT_USER)
        with st.sidebar:
            st.toast("Chat history cleared.", icon="🧹")

    st.subheader("Area of interest")

    # Create the Folium map centered on a predefined area
    m = folium.Map(location=[49.75, 13.39], zoom_start=13)

    draw = Draw(
        draw_options={
            'polyline': False,
            'polygon': False,
            'circle': False,
            'circlemarker': False,
            'marker': True, # Potentional hotel site
            'rectangle': True, # Area of interest
        },
        edit_options={
            'edit': True,
            'remove': True,
        },
        # Custom JS that ensures only one rectangle is present at a time on the map
        on={
            "add": JsCode(open("assets/js/handleAddDrawing.js", "r").read()),
            "remove": JsCode(open("assets/js/handleRemoveDrawing.js", "r").read()),
        }
    )
    draw.add_to(m)

    map_data = st_folium(
        m,
        width=400,
        height=500,
        key="map1",
        center=[49.75, 13.39],
        returned_objects=["all_drawings"],
        )
    
    st.session_state["selected_bbox"] = parse_drawing_coords(map_data, "Polygon")
    st.session_state["hotel_site_marker"] = parse_drawing_coords(map_data, "Point")

if prompt := st.chat_input(placeholder="Can you describe the selected area in terms of Open Land Use?"):
    if st.session_state["selected_bbox"] is None:
        st.toast("Please draw a rectangle on the map to select the area of interest.\n\
                 You can only have one area of interest selected at a time", icon="🗺️")
    else:
        coords = get_api_coords(st.session_state["selected_bbox"][0])
        hotel_site_marker = st.session_state["hotel_site_marker"]
        config = {"configurable": {"session_id": CHAT_USER}}

        app = build_graph()

        rewrite_chat_history()
        st.session_state["chat_prompted"] = True

        with st.spinner("Give me a second, I am thinking..."):
            last_message_id = 0
            for chunk in app.stream(
                {
                    "messages": [
                        SystemMessage(content=SYSTEM_MESSAGE_TEMPLATE),
                        HumanMessage(content=prompt)],
                    "coords": coords,
                    "hotel_site_marker": hotel_site_marker,
                },
                config=config,
                stream_mode="values"
            ):
                for i in range(last_message_id, len(chunk["messages"])):
                    message = chunk["messages"][i]
                    write_message(message)   
                last_message_id = len(chunk["messages"])

if not st.session_state["chat_prompted"]:
    rewrite_chat_history()