import streamlit as st
from utils.auth import initialize_auth
from utils.session import initialize_session_state
from components.sidebar import create_sidebar
from components.chat import display_chat_messages, process_user_input

from configuration.settings import APP_TITLE, APP_DESCRIPTION


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(page_title=APP_TITLE)
    
    # Initialize authentication
    hf_token = initialize_auth()
    
    if not hf_token:
        st.warning("Please enter valid API credentials to continue!", icon="⚠️")
        return
    
    # Set up sidebar with model selection and parameters
    model_config = create_sidebar()
    
    # Initialize session state for chat history
    initialize_session_state()
    
    # Display chat messages for the selected model
    display_chat_messages(model_config["model_name"])
    
    # Process user input when submitted
    process_user_input(model_config)

if __name__ == "__main__":
    main()