import json
import re
import time

import streamlit as st

from langchain_core.messages import AnyMessage
from langsmith import Client
from shapely.geometry import shape
from streamlit.components.v1 import html

from utils.agent_utils import get_chat_history

def parse_drawing_geometry(map_data: dict, drawing_type: str) -> str:
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
    return shape(last_drawing["geometry"]).wkt

def post_message_feedback(message: AnyMessage, key: str, correction: dict | None = None):
    # Every st widget with a defined key will be stored in the session state
    score = st.session_state.get(f"{message.run_id}_{getattr(message, "alternative_id", None)}")
    if score is None:
        return

    client = Client()
    if getattr(message, "feedback_score", None) is not None:
        if (score == message.feedback_score):
            return
        client.update_feedback(
            feedback_id=message.feedback_id,
            score=score
        )
    else:
        fb = client.create_feedback(
            run_id=message.run_id,
            key=key,
            score=score,
            correction=correction
        )
        message.feedback_id = fb.id
    message.feedback_score = score

def post_ab_feedback(run_id, value, comment, context_response_first):
    client = Client()
    client.create_feedback(
        run_id=run_id,
        key="ab",
        score=0 if value == "Option A" else 1,
        comment=comment,
        value=value,
        correction={"context_response": "Option A" if context_response_first else "Option B"}
    )

def print_tool_calls(tool_calls):
    texts = []
    for i, tc in enumerate(tool_calls, start=1):
        texts.append(f"**Tool Call {i}:** {tc['name']}  \n"\
            + f"Arguments: {json.dumps(tc['args'])}  \n"
        )
    return "\n".join(texts)

def write_message(message: AnyMessage):
    if message.type == "human":
        st.chat_message("human").write(message.content)
        return

    if message.type == "ai" and message.content != "":
        ai_msg = st.chat_message("ai", avatar="🌿")
        ai_msg.markdown(message.content.replace("\n", "  \n"), unsafe_allow_html=True)
        
        run_id = getattr(message, "run_id", None)
        if run_id is not None:
            ai_msg.feedback(
                "stars",
                key=f"{run_id}_{getattr(message, "alternative_id", None)}",
                on_change=post_message_feedback(message, "stars"),
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

def write_conversation():
    chat_history = get_chat_history()
    for m in chat_history.messages:
        alternative_id = getattr(m, "alternative_id", None)
        if alternative_id:
            alt_msg = st.session_state["all_messages"][alternative_id]
            write_comparison_messages(m, alt_msg)
        else:
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

def write_comparison_messages(main_msg, alt_msg):
    choice_clicked = getattr(main_msg, 'choice_clicked', None)
    # Ensure random message order because of A/B testing
    msg_A, msg_B = sorted([main_msg, alt_msg], key=lambda x: x.id.split('-')[1][0])
    element_id = f"{main_msg.id}_{alt_msg.id}"

    with st.chat_message("ai", avatar="🌿"):
        # User already selected preferred message, display only message present in history
        if choice_clicked:
            st.button(
                label="🔄 Swap preferred response",
                on_click=lambda: swap_preferred_message(main_msg, alt_msg),
                key=f"swap_btn_{element_id}"
            )
            st.markdown(main_msg.content.replace("\n", "  \n"), unsafe_allow_html=True)
            st.feedback(
                "stars",
                key=f"{main_msg.run_id}_{getattr(main_msg, 'alternative_id', None)}",
                on_change=post_message_feedback(main_msg, "stars", correction={"original_position": "Option A" if main_msg.id == msg_A.id else "Option B"}),
                disabled=st.session_state["inputs_disabled"]
            )
        else:
            col1, col2 = st.columns(2)
            col1.html(
                '''
                <div style="
                    background-color: #f0f2f6; 
                    padding: 10px; 
                    border-radius: 10px;
                    border: 2px solid #ddd;
                ">
                <strong>Option A</strong>
                '''
            )
            col1.markdown(msg_A.content.replace("\n", "  \n"))

            col2.html(
                '''
                <div style="
                    background-color: #f0f2f6; 
                    padding: 10px; 
                    border-radius: 10px;
                    border: 2px solid #ddd;
                ">
                <strong>Option B</strong>
                '''
            )
            col2.markdown(msg_B.content.replace("\n", "  \n"))

            # Feedback section
            # TODO - connect feedback to LangSmith
            with st.form(key=f"ab_form_{element_id}"):
                preferred_option = st.radio(
                    "**Which option do you prefer?**",
                    ["Option A", "Option B"],
                    index=None,
                    horizontal=True,
                    key=f"ab_radio_{element_id}"
                )
                feedback_text = st.text_input(
                    "Why do you like your selected option?",
                    key=f"fb_txt_{element_id}"
                )

                submit_button = st.form_submit_button("Submit")

                if submit_button:
                    if preferred_option:
                        st.success(f"Thanks for your feedback! You chose: {preferred_option}")
                        choose_preferred_message(main_msg, alt_msg, preferred_option)
                        post_ab_feedback(main_msg.run_id, preferred_option, feedback_text, msg_A.id == main_msg.id)
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.warning("Please select an option before submitting.")

def choose_preferred_message(main_msg, alt_msg, preferred_option):
    msg_A, _ = sorted([main_msg, alt_msg], key=lambda x: x.id.split('-')[1][0])
    main_msg.choice_clicked = True
    alt_msg.choice_clicked = True

    if (preferred_option == "Option A" and msg_A.id != main_msg.id)\
        or (preferred_option == "Option B" and msg_A.id == main_msg.id):
        swap_preferred_message(main_msg, alt_msg)

# TODO - preserve original message ids and other metadata after content swap?
def swap_preferred_message(main_msg, alt_msg):
    chat_history = get_chat_history()
    for i, m in enumerate(chat_history.messages):
        if m.id == main_msg.id:
            chat_history.messages[i] = alt_msg
    