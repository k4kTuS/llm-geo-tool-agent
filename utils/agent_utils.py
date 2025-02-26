from langchain_core.chat_history import InMemoryChatMessageHistory
import streamlit as st


def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if f'chat_history_{session_id}' not in st.session_state:
        st.session_state[f'chat_history_{session_id}'] = InMemoryChatMessageHistory()
    return st.session_state[f'chat_history_{session_id}']

def clear_chat_history(session_id: str):
    st.session_state[f'chat_history_{session_id}'] = InMemoryChatMessageHistory()