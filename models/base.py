"""Base model interface."""
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Base class for language models."""
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass