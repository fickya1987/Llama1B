"""Application configuration settings."""

# Application information
APP_TITLE = "💬 SLM-POC"
APP_DESCRIPTION = "This chatbot is created using various Small Language Models such as Llama 3.2, Gemma 2, Gemma 3, Phi 3.5, etc."



# Model definitions
MODELS = {

   "zephyr-7b-beta":  {
     "repo_id": "HuggingFaceH4/zephyr-7b-beta",
     "session_key" : "messages_zephyr"
    },
    
    "Llama 3.2 : 1B": {
        "repo_id": "meta-llama/Llama-3.2-1B-Instruct",
        "session_key": "messages_llama"
    },

    "Llama 3.2 : 3B": {

        "repo_id": "meta-llama/Llama-3.2-3B-Instruct",
        "session_key": "messages_llama_3B"

    },
    "Phi-3.5": {
        "repo_id": "microsoft/Phi-3.5-mini-instruct",
        "session_key": "messages_phi"
    },
    "Gemma 2 : 2B": {
        "repo_id": "google/gemma-2-2b-it",
        "session_key": "messages_gemma"
    },
    "Gemma 3 : 27B": {
        "repo_id": "google/gemma-3-27b-it",
        "session_key": "messages_gemma3"
    },

   "Qwen-32B" : {
      "repo_id" : "Qwen/QwQ-32B",
      "session_key" : "messages_qwen_32"
   },
}

# Default model parameters
DEFAULT_TEMPERATURE = 0.01
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_LENGTH = 2000

# Document processing settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# System prompts
DOCUMENT_QA_SYSTEM_PROMPT = """
Use the document to answer the questions.
If you don't know the answer, just say that you don't know. Do not generate other questions.
Use three sentences maximum and keep the answer as concise as possible.
"""

# Supported document types
SUPPORTED_DOC_TYPES = ["pdf", "docx", "txt"]
