"""Chat UI components."""
import streamlit as st
from models.huggingface import HuggingFaceModel
from utils.session import get_session_messages, add_message
from configuration.settings import DOCUMENT_QA_SYSTEM_PROMPT
from utils.document import cleanup_temp_file

def display_chat_messages(model_name):
    """Display chat messages for the selected model.
    
    Args:
        model_name (str): The name of the model
    """
    messages = get_session_messages(model_name)
    
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def process_user_input(model_config):
    """Process user input and generate a response.
    
    Args:
        model_config (dict): Configuration for the selected model
    """
    model_name = model_config["model_name"]
    document_data = model_config.get("document_data")
    
    # Get user input
    if prompt := st.chat_input():
        # Add user message to session and display
        add_message(model_name, "user", prompt)
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process with the selected model
        process_model_response(prompt, model_config)

def process_model_response(prompt, model_config):
    """Process the model response and display it.
    
    Args:
        prompt (str): The user's input
        model_config (dict): Configuration for the selected model
    """
    model_name = model_config["model_name"]
    repo_id = model_config["repo_id"]
    temperature = model_config["temperature"]
    top_p = model_config["top_p"]
    max_length = model_config["max_length"]
    document_data = model_config.get("document_data")
    
    # Initialize model
    model = HuggingFaceModel(repo_id)
    
    # Display special warning for Gemma 2
    if model_name == "Gemma 2 : 2B":
        st.write('⚠️ If you get Error: Clear chat history and try again.')
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if document_data is None:
                # Regular chat completion
                messages = get_session_messages(model_name)
                
                placeholder = st.empty()
                full_response = ""
                
                for chunk in model.generate_response(
                    messages, temperature, top_p, max_length
                ):
                    full_response += chunk
                    placeholder.markdown(full_response)
                
                if full_response.strip():
                    add_message(model_name, "assistant", full_response)
            else:
                # Document QA
                vector_store = document_data["vector_store"]
                
                full_response = model.answer_from_document(
                    prompt, vector_store, temperature, top_p, max_length
                )
                
                st.write(full_response)
                
                # Add system prompt and response to session
                add_message(model_name, "system", DOCUMENT_QA_SYSTEM_PROMPT)
                add_message(model_name, "assistant", full_response)
                
                # Clean up temp file after processing
                if "temp_file_path" in document_data:
                    cleanup_temp_file(document_data["temp_file_path"])