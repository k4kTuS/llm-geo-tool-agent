import configparser

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.language_models import BaseChatModel
import streamlit as st

from paths import PROJECT_ROOT

cfg = configparser.ConfigParser()
cfg.read(f'{PROJECT_ROOT}/config.ini')

# TODO - Add session id key for history, if we will store all the conversations during a session
def get_chat_history() -> InMemoryChatMessageHistory:
    if f'chat_history' not in st.session_state:
        st.session_state[f'chat_history'] = InMemoryChatMessageHistory()
    return st.session_state[f'chat_history']

def clear_chat_history():
    st.session_state[f'chat_history'] = InMemoryChatMessageHistory()

def get_llm() -> BaseChatModel:
    """
    Returns the LLM model based on project configuration. Currently supports OpenAI, Ollama and Groq.
    """
    provider = cfg['DEFAULT']['llm_provider']

    if provider == 'openai':
        return ChatOpenAI(
                    model=cfg['OPENAI']['model_id'],
                    temperature=0,
                    max_retries=2,
                )
    if provider == 'ollama':
        return ChatOllama(
                    model=cfg['OLLAMA']['model_id'],
                    temperature=0,
                    max_retries=2,
                )
    if provider == 'groq':
        return ChatGroq(
                    model=cfg['GROQ']['model_id'],
                    temperature=0,
                    max_retries=2,
                )
    raise ValueError(f"Unknown LLM provider: {provider}")