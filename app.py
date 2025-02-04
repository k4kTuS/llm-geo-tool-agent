import configparser
import time

from dotenv import load_dotenv
from langchain import callbacks
from langchain_core.messages import HumanMessage
import streamlit as st
from streamlit_folium import st_folium

from agents.lg_tool_agent import build_graph, clear_chat_history, SYSTEM_MESSAGE_TEMPLATE
from drawmap import DrawMap
from utils.streamlit_utils import *

load_dotenv()
cfg = configparser.ConfigParser()
cfg.read('config.ini')

if "message_to_run_ids" not in st.session_state:
    st.session_state["message_to_run_ids"] = {}
if "feedback_ids" not in st.session_state:
    st.session_state["feedback_ids"] = {}
if "inputs_disabled" not in st.session_state:
    st.session_state["inputs_disabled"] = False

st.set_page_config(
    page_title="PoliRuralPlus Chat Assistant",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="auto",
)

def disable_inputs():
    st.session_state["inputs_disabled"] = True

def show_login_form():
    st.title("Login")
    username_input = st.text_input("Enter your username")
    password_input = st.text_input("Enter the early access password", type="password")

    if st.button("Login"):
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
            clear_chat_history(st.session_state["user"])
            st.session_state["message_to_run_ids"] = {}
            st.session_state["feedback_ids"] = {}
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
        
        st.session_state["selected_bbox"] = parse_drawing_coords(map_data, "Polygon")
        st.session_state["hotel_site_marker"] = parse_drawing_coords(map_data, "Point")

    with st.expander("Click to select some example questions", icon="🔍"):
        examples = [line.rstrip() for line in open('resources/example_questions.txt')]
        selected = st.pills(label="What do you want to talk about?", options=examples, disabled=st.session_state["inputs_disabled"])
        add_pill_to_chat_input(selected)

    write_conversation()
    if prompt := st.chat_input(placeholder="Ask me anything...", disabled=st.session_state["inputs_disabled"], on_submit=disable_inputs):
        if st.session_state["selected_bbox"] is None:
            st.toast("Please draw a rectangle on the map to select the area of interest.", icon="🗺️")
            st.toast("You must have one area of interest selected at a time.", icon="🗺️")
            time.sleep(2)
        else:
            coords = get_api_coords(st.session_state["selected_bbox"][0])
            hotel_site_marker = st.session_state["hotel_site_marker"]
            config = {
                "configurable": {
                    "session_id": st.session_state["user"],
                },
                "metadata": {
                    "bounding-box": coords,
                    "user": st.session_state["user"],
                    "hotel_site_marker": hotel_site_marker 
                },
            }

            app = build_graph()
            with st.spinner("Give me a second, I am thinking..."):
                with callbacks.collect_runs() as cb:
                    last_message_id = 0
                    for chunk in app.stream(
                        {
                            "messages": [HumanMessage(content=prompt)],
                            "coords": coords,
                            "hotel_site_marker": hotel_site_marker,
                        },
                        config=config,
                        stream_mode="values",
                    ):
                        for i in range(last_message_id, len(chunk["messages"])):
                            message = chunk["messages"][i]
                            run_id = None
                            if (cb.traced_runs and cb.traced_runs[-1].name == cfg['LANGSMITH']['model_run_name']):
                                run_id = cb.traced_runs[-1].id
                            st.session_state["message_to_run_ids"][message.id] = run_id
                            write_message(message)
                        last_message_id = len(chunk["messages"])
        st.session_state["inputs_disabled"] = False
        st.rerun()

if "user" not in st.session_state:
    show_login_form()
else:
    show_chat_app()