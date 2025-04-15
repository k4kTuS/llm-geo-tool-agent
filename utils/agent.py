from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, AIMessage
import streamlit as st

LLM_OPTIONS = ['gpt-4o-mini', 'deepseek-r1-distill-llama-70b', 'llama3.2:3b-instruct-fp16', 'llama3.2']
LLM_PROVIDERS = {
    'gpt-4o-mini': 'openai',
    'deepseek-r1-distill-llama-70b': 'groq',
    'llama3.2:3b-instruct-fp16': 'ollama',
    'llama3.2': 'ollama',
}
DEFAULT_LLM = 'gpt-4o-mini'

def get_chat_history(alternative: bool = False) -> InMemoryChatMessageHistory:
    chat_key = f'chat_{st.session_state.thread_id}' if not alternative else f'chat_alt_{st.session_state.thread_id}'
    if chat_key not in st.session_state:
        st.session_state[chat_key] = InMemoryChatMessageHistory()
    return st.session_state[chat_key]

def clear_chat_history():
    chat_key = f'chat_{st.session_state.thread_id}'
    chat_alt_key = f'chat_alt_{st.session_state.thread_id}'
    if chat_key in st.session_state:
        del st.session_state[chat_key]
    if chat_alt_key in st.session_state:
        del st.session_state[chat_alt_key]
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

def store_run_messages(messages: list[AnyMessage], alternative: bool = False):
    """
    Add all messages to the chat history and store each message in the session state.
    """
    get_chat_history(alternative=alternative).add_messages(messages)
    for message in messages:
        st.session_state.all_messages[message.id] = message

def pair_response_messages(run_id: str, main: AIMessage, alternative: AIMessage):
    """
    Set the metadata for the main and alternative messages to link them together.
    """
    main.alternative_id = alternative.id
    alternative.alternative_id = main.id
    alternative.run_id = run_id