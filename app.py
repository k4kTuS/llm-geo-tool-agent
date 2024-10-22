import configparser
import os
import folium
import streamlit as st

from langchain_core.messages import HumanMessage, SystemMessage

from streamlit_folium import st_folium
from agent import build_graph, get_chat_history, clear_chat_history, SYSTEM_MESSAGE_TEMPLATE

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
        st.markdown(message.pretty_repr().replace("\n", "  \n"), unsafe_allow_html=True)
    elif message.type == "ai" and not has_tool_info:
        st.chat_message("ai").markdown(message.content.replace("\n", "  \n"), unsafe_allow_html=True)

def rewrite_chat_history():
    chat_history = get_chat_history(CHAT_USER)
    for m in chat_history.messages:
        write_message(m)

st.title("PoliRuralPlus Chat Assistant")

st.session_state["chat_prompted"] = False

with st.sidebar as sidebar:
    st.subheader("Chat management")
    
    switch = st.checkbox("Show tool calls details")

    if switch:
        st.session_state["show_tool_calls"] = True
    else:
        st.session_state["show_tool_calls"] = False

    if st.button("Clear chat history"):
        clear_chat_history(CHAT_USER)
        with st.sidebar:
            st.info("Chat history cleared.")

    st.subheader("Area of interest")

    # Create the Folium map centered on the highlighted area
    m = folium.Map(location=[49.75, 13.39], zoom_start=13)

    # Define bounding box styles for the selected area
    kw_bb = {
        "color": "blue",
        "weight": 2,
        "fill": True,
        "fill_color": "blue",
        "fillOpacity": 0.15,
        "tooltip": "<strong>Selected map area</strong>",
    }

    # Add the selected area rectangle to the map
    folium.Rectangle(
        bounds=[[49.72, 13.35], [49.77, 13.4]],
        **kw_bb
    ).add_to(m)

    # Define bounding box styles for the highlighted area
    kw_hs = {
        "color": "red",
        "weight": 2,
        "fill": True,
        "fill_color": "red",
        "fillOpacity": 0.15,
        "tooltip": "<strong>Highlighted hotels area</strong>",
    }

    # Add the highlighted area rectangle to the map
    folium.Rectangle(
        bounds=[[49.74, 13.36], [49.76, 13.38]],
        **kw_hs
    ).add_to(m)

    # Use st_folium to display the map with specific width and height
    st_folium(m, width=400, height=500, returned_objects=[], center=[49.745, 13.375])

if prompt := st.chat_input(placeholder="Can you describe the selected area in terms of Open Land Use?"):
    coords = [49.72,13.35,49.77,13.4]
    highlighted_square = [49.74, 13.36, 49.76, 13.38]
    config = {"configurable": {"session_id": CHAT_USER}}

    app = build_graph()

    rewrite_chat_history()
    st.session_state["chat_prompted"] = True

    last_message_id = 0
    for chunk in app.stream(
        {
            "messages": [
                SystemMessage(content=SYSTEM_MESSAGE_TEMPLATE),
                HumanMessage(content=prompt)],
            "coords": coords,
            "highlighted_square": highlighted_square,
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