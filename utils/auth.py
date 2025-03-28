"""Authentication utilities for the application."""
import os
import streamlit as st
from dotenv import load_dotenv

def initialize_auth():
    """Initialize authentication and return the HuggingFace token.
    
    Returns:
        str: The Hugging Face API token or None if not provided
    """
    load_dotenv()
    
    with st.sidebar:
        # Try to get token from environment variables or secrets
        if os.getenv('HF_TOKEN') is not None:
            st.success('API key already provided!', icon='✅')
            hf_token = os.getenv('HF_TOKEN')
        elif 'HF_TOKEN' in st.secrets:
            st.success('API key already provided!', icon='✅')
            hf_token = st.secrets['HF_TOKEN']
            os.environ['HF_TOKEN'] = hf_token
        else:
            # If not available, request from user
            hf_token = st.text_input('Enter HF API token:', type='password')
            if hf_token:
                os.environ['HF_TOKEN'] = hf_token
            
        # Validate token
        if hf_token and not hf_token.startswith('hf_'):
            st.warning('Please enter a valid Hugging Face API token!', icon='⚠️')
            return None
        elif hf_token:
            st.success('Proceed to entering your prompt message!', icon='👉')
            return hf_token
            
    return hf_token