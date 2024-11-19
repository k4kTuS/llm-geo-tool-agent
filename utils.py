import json
import re

import streamlit as st

from langchain_core.messages import BaseMessage
from langsmith import Client
from streamlit.components.v1 import html

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

def post_feedback(run_id):
    # Every st widget with a defined key will be stored in the session state
    if run_id not in st.session_state or st.session_state[run_id] is None:
        return

    client = Client()
    if run_id in st.session_state["feedback_ids"]:
        if (st.session_state[run_id] == st.session_state["feedback_ids"][run_id]):
            return
        client.update_feedback(
            run_id,
            score=st.session_state[run_id]
        )
    else:
        client.create_feedback(
            run_id,
            key="thumbs",
            score=st.session_state[run_id],
            feedback_id=run_id
        )
    st.session_state["feedback_ids"][run_id] = st.session_state[run_id]

def print_tool_calls(tool_calls):
    texts = []
    for i, tc in enumerate(tool_calls, start=1):
        texts.append(f"**Tool Call {i}:** {tc['name']}  \n"\
            + f"Arguments: {json.dumps(tc['args'])}  \n"
        )
    return "\n".join(texts)

def write_message(message: BaseMessage):
    if message.type == "human":
        st.chat_message("human").write(message.content)
        return

    if message.type == "ai" and message.content != "":
        ai_msg = st.chat_message("ai", avatar="🌿")
        ai_msg.markdown(message.content.replace("\n", "  \n"), unsafe_allow_html=True)
        run_id = st.session_state["message_to_run_ids"].get(message.id)
        if run_id is not None:
            ai_msg.feedback(
                "thumbs",
                key=run_id,
                on_change=post_feedback(run_id),
                disabled=st.session_state["inputs_disabled"]
            )

    if st.session_state["show_tool_calls"]:
        if "tool_calls" in message.additional_kwargs:
            gen_tools_msg = st.chat_message("tool", avatar="🛠️")
            with gen_tools_msg.expander("Generated tool calls:"):
                st.markdown(print_tool_calls(message.tool_calls))
        elif message.type == "tool":
            tool_msg = st.chat_message("tool", avatar="🛠️")
            with tool_msg.expander(message.name):
                st.markdown(message.content.replace("\n", "  \n"), unsafe_allow_html=True)

def rewrite_chat_history():
    chat_history = get_chat_history(st.session_state["user"])
    for m in chat_history.messages:
        write_message(m)

def add_pill_to_chat_input(pill):
    # Remove the emoji from the pill
    if pill is not None:
        pill = re.sub(r'^:[a-z_]+: ', '', pill)

    js_pill = f'"{pill}"' if pill else "null"
    js = f"""
        <script>
            function insertText(dummy_var_to_force_repeat_execution) {{
                var chatInput = parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
                var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                nativeInputValueSetter.call(chatInput, {js_pill});
                var event = new Event('input', {{bubbles: true}}); 
                chatInput.dispatchEvent(event);
            }}
            insertText(42);
        </script>
        """
    html(js, height=0, width=0)