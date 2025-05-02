import time
import uuid

from dotenv import load_dotenv
import streamlit as st
from streamlit_folium import st_folium

from agents import get_agent, BaseAgent
from config.project_paths import RESOURCES_DIR
from visualizations.drawmap import DrawMap
from utils.streamlit import (
    add_pill_to_chat_input,
    initialize_custom_css,
    initialize_session_state,
    parse_drawing_geometry,
    write_conversation,
    write_message
)
from schemas.geometry import BoundingBox, PointMarker
from utils.agent import clear_chat_history, LLM_OPTIONS

load_dotenv()
initialize_session_state()

st.set_page_config(
    page_title="GeoChat Assistant",
    page_icon="🌿",
    layout="wide" if "user" in st.session_state else "centered",
    initial_sidebar_state="auto",
)
initialize_custom_css()

def disable_inputs():
    st.session_state.inputs_disabled = True

def show_login_form():
    st.title("Login")

    with st.form("login_form"):
        username_input = st.text_input("Enter your username")
        password_input = st.text_input("Enter the early access password", type="password")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            if not username_input:
                st.warning("Please enter your username.")
            elif password_input != st.secrets.EA_PASSWORD:
                st.error("Incorrect password. Please try again.")
            else:
                st.session_state.user = username_input
                st.success(f"Welcome, {username_input}! Redirecting you to chat assistant...")
                time.sleep(1)
                st.rerun()

def show_chat_app():
    with st.sidebar:
        st.title("🌿 GeoChat Assistant")
        
        with st.popover("Settings", icon="⚙️", use_container_width=True):
            st.toggle(
                label="Show tool calls",
                value=False,
                disabled=st.session_state.inputs_disabled,
                on_change=lambda: st.session_state.update({"show_tool_calls": not st.session_state.show_tool_calls})
            )

            st.warning("Changing the LLM model will clear the chat history.")
            llm_selection_locked = st.secrets.get("LOCK_LLM_SELECTION", False)
            selected_llm = st.selectbox(
                label="LLM model - Selection locked" if llm_selection_locked else "LLM model",
                options=LLM_OPTIONS,
                disabled=True if llm_selection_locked else st.session_state.inputs_disabled,
            )
            if selected_llm != st.session_state.llm_choice:
                st.toast(f"Changed LLM model to {selected_llm}.", icon="🔄")
                st.session_state.llm_choice = selected_llm
                clear_chat_history()
                st.session_state.thread_id = uuid.uuid4()
                st.toast("Chat history cleared.", icon="🧹")

        st.header("Chat management")
        if st.button(label="Clear chat history", disabled=st.session_state.inputs_disabled, use_container_width=True):
            clear_chat_history()
            st.session_state.thread_id = uuid.uuid4()
            st.toast("Chat history cleared.", icon="🧹")

        st.header("Select an area of interest")
        m = DrawMap()
        map_data = st_folium(
            m.map_,
            use_container_width=True,
            height=500,
            key="map1",
            center=[49.75, 13.39],
            returned_objects=["all_drawings"],
        )
        
        st.session_state.selected_area_wkt = parse_drawing_geometry(map_data, "Polygon")
        st.session_state.point_marker_wkt = parse_drawing_geometry(map_data, "Point")

    with st.expander("Click to select some example questions", icon="🔍"):
        examples = [line.rstrip() for line in open(f'{RESOURCES_DIR}/example_questions.txt')]
        selected = st.pills(label="What do you want to talk about?", options=examples, disabled=st.session_state.inputs_disabled)
        add_pill_to_chat_input(selected)

    write_conversation()
    if prompt := st.chat_input(placeholder="Ask me anything...", disabled=st.session_state.inputs_disabled, on_submit=disable_inputs):
        if st.session_state.selected_area_wkt is None:
            st.toast("Please draw a rectangle on the map to select the area of interest.", icon="🗺️")
            st.toast("You must have one area of interest selected at a time.", icon="🗺️")
            time.sleep(2)
        else:
            bbox = BoundingBox(wkt=st.session_state.selected_area_wkt, crs="EPSG:4326")
            point_marker = PointMarker(wkt=st.session_state.point_marker_wkt, crs="EPSG:4326") if st.session_state.point_marker_wkt else None
            
            run_id = uuid.uuid4() # For langsmith
            config = {
                "run_id": run_id,
                "configurable": {
                    "run_id": run_id, # Used for feedback, accessible from graph nodes
                    "session_id": f"{st.session_state.user}-{st.session_state.thread_id}", # Used to group traces in langsmith
                    "model_name": st.session_state.llm_choice
                },
                "metadata": {
                    "bounding_box_wkt": bbox.wkt,
                    "user": st.session_state.user,
                    "point_marker": point_marker.wkt if point_marker else None,
                },
            }

            agent: BaseAgent = get_agent(name="comparison_geo")
            with st.spinner("Give me a second, I am thinking..."):
                last_message_id = 0
                state = None
                for chunk in agent.run(prompt, bbox, point_marker, config):
                    if "selected_tools" in chunk:
                        st.session_state.filtered_tools = chunk["selected_tools"]
                    for i in range(last_message_id, len(chunk["messages"])):
                        message = chunk["messages"][i]
                        write_message(message)
                    last_message_id = len(chunk["messages"])
                    state = chunk
                agent.store_run_details(state, config)
        st.session_state.inputs_disabled = False
        st.rerun()

if "user" not in st.session_state:
    show_login_form()
else:
    show_chat_app()