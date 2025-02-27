from langchain_core.chat_history import InMemoryChatMessageHistory
import streamlit as st

# TODO - Add session id key for history, if we will store all the conversations during a session
def get_chat_history() -> InMemoryChatMessageHistory:
    if f'chat_history' not in st.session_state:
        st.session_state[f'chat_history'] = InMemoryChatMessageHistory()
    return st.session_state[f'chat_history']

def clear_chat_history():
    st.session_state[f'chat_history'] = InMemoryChatMessageHistory()