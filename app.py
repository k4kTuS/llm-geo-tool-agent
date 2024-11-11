import time

import folium
from folium.plugins import Draw
from folium.utilities import JsCode
from langchain_core.messages import HumanMessage, SystemMessage
import streamlit as st
from streamlit_folium import st_folium

from agent import build_graph, clear_chat_history, SYSTEM_MESSAGE_TEMPLATE
from utils import *

st.set_page_config(
    page_title="PoliRuralPlus Chat Assistant",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="auto",
)

def show_login_form():
    st.title("Login")
    username_input = st.text_input("Enter your username")

    if st.button("Login") and username_input:
        st.session_state["user"] = username_input
        st.success(f"Welcome, {username_input}! Redirecting you to chat assistant...")
        time.sleep(1)
        st.rerun()

def show_chat_app():
    st.title("🌿 PoliRuralPlus Chat Assistant")

    st.session_state["chat_prompted"] = False

    with st.sidebar:
        st.subheader("Chat management")
        
        tools_on = st.toggle("Show tool calls details")

        if tools_on:
            st.session_state["show_tool_calls"] = True
        else:
            st.session_state["show_tool_calls"] = False

        if st.button("Clear chat history"):
            clear_chat_history(st.session_state["user"])
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
            st.toast("Please draw a rectangle on the map to select the area of interest.", icon="🗺️")
            st.toast("You must have one area of interest selected at a time.", icon="🗺️")
        else:
            coords = get_api_coords(st.session_state["selected_bbox"][0])
            hotel_site_marker = st.session_state["hotel_site_marker"]
            config = {"configurable": {"session_id": st.session_state["user"]}}

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

if "user" not in st.session_state:
    show_login_form()
else:
    show_chat_app()