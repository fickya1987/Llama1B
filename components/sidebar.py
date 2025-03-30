"""Sidebar UI components."""
import os
import streamlit as st
from utils.session import clear_messages
from utils.document import load_document, process_documents, cleanup_temp_file
from configuration.settings import APP_DESCRIPTION, MODELS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_MAX_LENGTH, SUPPORTED_DOC_TYPES

def create_sidebar():
    """Create and render the sidebar UI.
    
    Returns:
        dict: Configuration for the selected model
    """
    with st.sidebar:
        st.title('💬 Small Language Models - POC')
        st.write(APP_DESCRIPTION)
        
        # Model selection
        model_name = st.selectbox(
            "Select the Model:",
            list(MODELS.keys())
        )
        
        # Model parameters
        st.subheader('Model parameters')
        temperature = st.slider('Temperature', min_value=0.01, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.01)
        top_p = st.slider('Top P', min_value=0.01, max_value=1.0, value=DEFAULT_TOP_P, step=0.01)
        max_length = st.slider('Max Length', min_value=20, max_value=2040, value=DEFAULT_MAX_LENGTH, step=5)
        
        # Clear chat button
        if st.button('Clear Chat History'):
            clear_messages(model_name)
            st.rerun()
        
        # File uploader
        document_data = handle_document_upload()
        
        model_config = {
            "model_name": model_name,
            "repo_id": MODELS[model_name]["repo_id"],
            "temperature": temperature,
            "top_p": top_p,
            "max_length": max_length,
            "document_data": document_data
        }
        
        return model_config

def handle_document_upload():
    """Handle document upload in the sidebar.
    
    Returns:
        dict: Document data if a document was uploaded and processed, None otherwise
    """
    st.subheader("Document Upload")
    uploaded_file = st.file_uploader(
        f"Upload a document ({', '.join(SUPPORTED_DOC_TYPES)})", 
        type=SUPPORTED_DOC_TYPES
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner("Processing document..."):
                documents, temp_file_path = load_document(uploaded_file)
                vector_store = process_documents(documents, os.getenv('HF_TOKEN'))
                
                st.success("Document uploaded and processed successfully!")
                
                return {
                    "vector_store": vector_store,
                    "temp_file_path": temp_file_path,
                    "filename": uploaded_file.name
                }
        except Exception as e:
            st.error(f"Error processing document: {e}")
            return None
    
    return None