"""Document processing utilities."""
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import FAISS
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

def load_document(uploaded_file):
    """Load a document from an uploaded file.
    
    Args:
        uploaded_file (UploadedFile): The uploaded file object
        
    Returns:
        tuple: (documents, temp_file_path) where documents are the loaded documents and 
               temp_file_path is the path to the temporary file
    """
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    
    # Determine loader based on file type
    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(temp_file_path)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        loader = Docx2txtLoader(temp_file_path)
    else:  # Default to text loader
        loader = TextLoader(temp_file_path)
    
    documents = loader.load()
    return documents, temp_file_path

def process_documents(documents, hf_token):
    """Process documents for retrieval.
    
    Args:
        documents (list): List of loaded documents
        hf_token (str): Hugging Face API token
        
    Returns:
        FAISS: Vector store for document retrieval
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceHubEmbeddings(
        huggingfacehub_api_token=hf_token
    )
    
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

def cleanup_temp_file(file_path):
    """Clean up temporary file after processing.
    
    Args:
        file_path (str): Path to the temporary file
    """
    try:
        os.unlink(file_path)
    except Exception as e:
        # Log error but don't fail if cleanup fails
        print(f"Error cleaning up temporary file: {e}")