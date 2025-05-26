"""
model_config.py - Configuration for LLM and embedding models
"""
import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from llama_index.llms.openai import OpenAI

# Import config file
# For common environment: from config import config
# For restricted environment: from config_restricted import config
from config import config  # Change this line based on your environment

class ModelConfig:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize model configuration
        
        Args:
            model_name: Name of the LLM model to use (default: "gpt-3.5-turbo")
        """
        self.model_name = model_name
        self.embedding_model = config["embedding_model"]
        self.embedding_func = None
        self.llm = None
        self.chroma_client = None
        self.chat_sessions = None
        self.doc_qa = None
        
        # Initialize all components
        self._init_all()
    
    def _init_all(self):
        """Initialize all components"""
        try:
            # Initialize embedding function
            self.embedding_func = OpenAIEmbeddingFunction(
                api_key=config["api_key"],
                api_base=config["api_base"],
                model_name=self.embedding_model
            )
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=config["chroma_path"]
            )
            
            # Create collections
            self.chat_sessions = self.chroma_client.get_or_create_collection(
                name="chat_sessions",
                embedding_function=self.embedding_func
            )
            
            self.doc_qa = self.chroma_client.get_or_create_collection(
                name="doc_qa",
                embedding_function=self.embedding_func
            )
            
            # Initialize LLM
            self.llm = OpenAI(
                model=self.model_name,
                api_key=config["api_key"],
                base_url=config["api_base"]
            )
            
        except Exception as e:
            print(f"Error initializing model configuration: {e}")
            raise
    
    def _init_collections(self):
        """Initialize ChromaDB collections with proper embedding functions"""
        try:
            # Chat sessions collection
            self.chat_sessions = self.chroma_client.get_or_create_collection(
                name="chat_sessions",
                embedding_function=self.embedding_func
            )
            
            # Document QA collection
            self.doc_qa = self.chroma_client.get_or_create_collection(
                name="doc_qa",
                embedding_function=self.embedding_func
            )
        except Exception as e:
            print(f"Error initializing collections: {e}")
            raise
