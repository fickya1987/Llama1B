"""HuggingFace model implementations."""
import os
from huggingface_hub import InferenceClient
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceEndpoint
from models.base import BaseModel
from configuration.settings import DOCUMENT_QA_SYSTEM_PROMPT

class HuggingFaceModel(BaseModel):
    """HuggingFace model implementation."""
    
    def __init__(self, repo_id):
        """Initialize the model.
        
        Args:
            repo_id (str): The repository ID for the model
        """
        self.repo_id = repo_id
        self.client = InferenceClient(api_key=os.getenv('HF_TOKEN'))
    
    def generate_response(self, messages, temperature, top_p, max_length):
        """Generate a response from the model.
        
        Args:
            messages (list): List of message objects
            temperature (float): Temperature parameter for generation
            top_p (float): Top-p parameter for generation
            max_length (int): Maximum length of the generated response
            
        Returns:
            str: The generated response
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.repo_id,
                messages=messages,
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                stream=True
            )
            
            full_response = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    full_response += chunk.choices[0].delta.content
            
            return full_response
        except Exception as e:
            # Log the error and return a friendly message
            print(f"Error generating response from {self.repo_id}: {e}")
            return f"I'm sorry, there was an error generating a response. Please try again or choose a different model."
    
    def answer_from_document(self, query, vector_store, temperature, top_p, max_length):
        """Answer a query using document retrieval.
        
        Args:
            query (str): The user's query
            vector_store (VectorStore): Vector store for document retrieval
            temperature (float): Temperature parameter for generation
            top_p (float): Top-p parameter for generation
            max_length (int): Maximum length of the generated response
            
        Returns:
            str: The generated response
        """
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=HuggingFaceEndpoint(
                    repo_id=self.repo_id,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=5,
                    huggingfacehub_api_token=os.getenv('HF_TOKEN'),
                ),
                chain_type="stuff",
                retriever=vector_store.as_retriever()
            )
            
            response = qa_chain({"query": query})
            return response["result"]
        except Exception as e:
            # Log the error and return a friendly message
            print(f"Error retrieving answer from document with {self.repo_id}: {e}")
            return "I'm sorry, there was an error processing your document query. Please try again or choose a different model."
        
