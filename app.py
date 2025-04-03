import configparser
import time
import uuid

from dotenv import load_dotenv
from langchain import callbacks
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
import streamlit as st
from streamlit_folium import st_folium

from agents.geo_agent import geo_agent
from agents.comparison_geo_agent import comparison_geo_agent
from visualizations.drawmap import DrawMap
from paths import PROJECT_ROOT
from utils.streamlit_utils import *
from schemas.geometry import BoundingBox, PointMarker
from utils.agent_utils import clear_chat_history

load_dotenv()
cfg = configparser.ConfigParser()
cfg.read(f'{PROJECT_ROOT}/config.ini')

if "inputs_disabled" not in st.session_state:
    st.session_state["inputs_disabled"] = False
if "all_messages" not in st.session_state:
    st.session_state["all_messages"] = {}

st.set_page_config(
    page_title="PoliRuralPlus Chat Assistant",
    page_icon="🌿",
    layout="wide" if "user" in st.session_state else "centered",
    initial_sidebar_state="auto",
)

def disable_inputs():
    st.session_state["inputs_disabled"] = True

def show_login_form():
    st.title("Login")

    with st.form("login_form"):
        username_input = st.text_input("Enter your username")
        password_input = st.text_input("Enter the early access password", type="password")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            if not username_input:
                st.warning("Please enter your username.")
            elif password_input != st.secrets["EA_PASSWORD"]:
                st.error("Incorrect password. Please try again.")
            else:
                st.session_state["user"] = username_input
                st.success(f"Welcome, {username_input}! Redirecting you to chat assistant...")
                time.sleep(1)
                st.rerun()

def show_chat_app():
    st.title("🌿 PoliRuralPlus Chat Assistant")

    with st.sidebar:
        st.subheader("Chat management")
        
        tools_on = st.toggle(label="Show tool calls details", disabled=st.session_state["inputs_disabled"])

        if tools_on:
            st.session_state["show_tool_calls"] = True
        else:
            st.session_state["show_tool_calls"] = False

        if st.button(label="Clear chat history", disabled=st.session_state["inputs_disabled"]):
            clear_chat_history()
            st.toast("Chat history cleared.", icon="🧹")

        st.subheader("Area of interest")

        m = DrawMap()
        map_data = st_folium(
            m.map_,
            width=400,
            height=500,
            key="map1",
            center=[49.75, 13.39],
            returned_objects=["all_drawings"],
        )
        
        st.session_state["selected_area_wkt"] = parse_drawing_geometry(map_data, "Polygon")
        st.session_state["hotel_site_wkt"] = parse_drawing_geometry(map_data, "Point")

    with st.expander("Click to select some example questions", icon="🔍"):
        examples = [line.rstrip() for line in open('resources/example_questions.txt')]
        selected = st.pills(label="What do you want to talk about?", options=examples, disabled=st.session_state["inputs_disabled"])
        add_pill_to_chat_input(selected)

    write_conversation()
    if prompt := st.chat_input(placeholder="Ask me anything...", disabled=st.session_state["inputs_disabled"], on_submit=disable_inputs):
        if st.session_state["selected_area_wkt"] is None:
            st.toast("Please draw a rectangle on the map to select the area of interest.", icon="🗺️")
            st.toast("You must have one area of interest selected at a time.", icon="🗺️")
            time.sleep(2)
        else:
            bbox = BoundingBox(wkt=st.session_state["selected_area_wkt"])
            hotel_site_marker = PointMarker(wkt=st.session_state["hotel_site_wkt"]) if st.session_state["hotel_site_wkt"] else None
            
            run_id = uuid.uuid4() # For langsmith
            config = {
                "run_id": run_id,
                "configurable": {
                    "run_id": run_id, # Used for feedback, accessible from graph nodes
                },
                "metadata": {
                    "bounding_box_wkt": bbox.wkt,
                    "user": st.session_state["user"],
                    "hotel_site_marker_wkt": hotel_site_marker.wkt if hotel_site_marker else None,
                },
            }
            input = {
                "messages": [HumanMessage(content=prompt)],
                "bounding_box": bbox,
                "hotel_site_marker": hotel_site_marker,
            }

            agent: CompiledStateGraph = comparison_geo_agent
            with st.spinner("Give me a second, I am thinking..."):
                last_message_id = 0
                for chunk in agent.stream(
                    input=input,
                    config=config,
                    stream_mode="values",
                ):
                    for i in range(last_message_id, len(chunk["messages"])):
                        message = chunk["messages"][i]
                        write_message(message)
                    last_message_id = len(chunk["messages"])
                
        st.session_state["inputs_disabled"] = False
        st.rerun()

if "user" not in st.session_state:
    show_login_form()
else:
    show_chat_app()