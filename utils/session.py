"""Session state management utilities."""
import streamlit as st
from configuration.settings import MODELS

def initialize_session_state():
    """Initialize session state for all models if not already set."""
    for model_name, model_config in MODELS.items():
        session_key = model_config["session_key"]
        
        if session_key not in st.session_state:
            st.session_state[session_key] = []
            st.session_state[session_key] = [{"role": "assistant", "content": "How may I assist you today?"}]

def get_session_messages(model_name):
    """Get session messages for the specified model.
    
    Args:
        model_name (str): The name of the model
        
    Returns:
        list: List of message objects for the specified model
    """
    session_key = MODELS[model_name]["session_key"]
    return st.session_state[session_key]

def add_message(model_name, role, content):
    """Add a message to the session state for the specified model.
    
    Args:
        model_name (str): The name of the model
        role (str): The role of the message sender (user, assistant, system)
        content (str): The content of the message
    """
    session_key = MODELS[model_name]["session_key"]
    st.session_state[session_key].append({"role": role, "content": content})

def clear_messages(model_name):
    """Clear messages for the specified model.
    
    Args:
        model_name (str): The name of the model
    """
    session_key = MODELS[model_name]["session_key"]
    st.session_state[session_key] = []
    st.session_state[session_key] = [{"role": "assistant", "content": "How may I assist you today?"}]