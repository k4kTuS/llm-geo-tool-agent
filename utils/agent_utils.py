from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.language_models import BaseChatModel
import streamlit as st

LLM_OPTIONS = ['gpt-4o-mini', 'deepseek-r1-distill-llama-70b', 'llama3.2:3b-instruct-fp16', 'llama3.2']
LLM_PROVIDERS = {
    'gpt-4o-mini': 'openai',
    'deepseek-r1-distill-llama-70b': 'groq',
    'llama3.2:3b-instruct-fp16': 'ollama',
    'llama3.2': 'ollama',
}
DEFAULT_LLM = 'gpt-4o-mini'

# TODO - Add session id key for history, if we will store all the conversations during a session
def get_chat_history() -> InMemoryChatMessageHistory:
    if f'chat_history' not in st.session_state:
        st.session_state.chat_history = InMemoryChatMessageHistory()
    return st.session_state.chat_history

def clear_chat_history():
    st.session_state.chat_history = InMemoryChatMessageHistory()
    st.session_state.all_messages = {}

def get_llm(model_name: str) -> BaseChatModel:
    """
    Returns the LLM model based on project configuration. Currently supports OpenAI, Ollama and Groq.
    """
    provider = LLM_PROVIDERS.get(model_name, None)
    if provider == 'openai':
        return ChatOpenAI(
                    model=model_name,
                    temperature=0,
                    max_retries=2,
                )
    if provider == 'ollama':
        return ChatOllama(
                    model=model_name,
                    temperature=0,
                    max_retries=2,
                )
    if provider == 'groq':
        return ChatGroq(
                    model=model_name,
                    temperature=0,
                    max_retries=2,
                )
    raise ValueError(f"Unknown LLM provider for model: {model_name}")