import os
import sys
from typing import Dict, Any

class Config:
    
    # Document Procesing
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    DEFAULT_ENCODING = 'utf-8'

    # Embedding Model
    DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2' 

    # Vector Store
    DEFAULT_PERSIST_DIRECTORY = "./chroma_db"
    DEFAULT_RETRIEVAL_K = 5 

    # LLM Settings
    DEFAULT_TEMPERATURE = 0.3 
    DEFAULT_CHAIN_TYPE = "stuff" 

    #File Settings
    SUPPORTED_FILE_TYPES = ["txt"]
    MAX_FILE_SIZE_MB = 100

    @classmethod 
    def get_doc_processing_config(cls) -> Dict[str, Any]:
        return {
            'chunk_size': int(os.getenv('CHUNK_SIZE', cls.DEFAULT_CHUNK_SIZE)),
            'chunk_overlap': int(os.getenv('CHUNK_OVERLAP', cls.DEFAULT_CHUNK_OVERLAP)),
            'encoding': os.getenv('ENCODING', cls.DEFAULT_ENCODING)
        }
    
    @classmethod
    def get_embedding_config(cls) -> Dict[str, Any]:
        return {
            'model_name': os.getenv('EMBEDDING_MODEL', cls.DEFAULT_EMBEDDING_MODEL),
        }
    
    @classmethod
    def get_vector_store_config(cls) -> Dict[str, Any]:
        return {
            'persist_directory': os.getenv('PERSIST_DIRECTORY', cls.DEFAULT_PERSIST_DIRECTORY),
            'retrieval_k': int(os.getenv('RETRIEVAL_K', cls.DEFAULT_RETRIEVAL_K))
        }
    
    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        return {
            'temperature': float(os.getenv('LLM_TEMPERATURE', cls.DEFAULT_TEMPERATURE)),
            'chain_type': os.getenv('LLM_CHAIN_TYPE', cls.DEFAULT_CHAIN_TYPE),
            'api_key': os.getenv('GOOGLE_API_KEY')
        }
    
    @classmethod
    def get_file_settings(cls) -> Dict[str, Any]:
        return {
            'supported_types': cls.SUPPORTED_FILE_TYPES,
            'max_size_mb': int(os.getenv('MAX_FILE_SIZE_MB', cls.MAX_FILE_SIZE_MB))
        }
    
    @classmethod
    def get_all_configs(cls) -> Dict[str, Any]:
        return {
            'document_processing': cls.get_doc_processing_config(),
            'embedding': cls.get_embedding_config(),
            'vector_store': cls.get_vector_store_config(),
            'llm': cls.get_llm_config(),
            'file_settings': cls.get_file_settings()
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        llm_config = cls.get_llm_config()

        if not llm_config['api_key']:
            return False
        
        return True
    
    @classmethod
    def get_environment_info(cls) -> Dict[str, Any]: 
        return {
            'python_version': sys.version,
            'environment_variables': {
                'GOOGLE_API_KEY': 'SET' if os.getenv('GOOGLE_API_KEY') else 'NOT SET',
                'CHUNK_SIZE': os.getenv('CHUNK_SIZE', 'DEFAULT'),
                'EMBEDDING_MODEL': os.getenv('EMBEDDING_MODEL', 'DEFAULT'),
                'PERSIST_DIRECTORY': os.getenv('PERSIST_DIRECTORY', 'DEFAULT'),
            }
        }


        

        