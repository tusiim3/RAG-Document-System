import logging
import os
from typing import List, Optional, Tuple
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, persist_directory: str = "./chroma_db", embedding_function: Optional[Embeddings] = None):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.vector_store = None
        self._ensure_persist_directory()

    def _ensure_persist_directory(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            logger.info(f"Persist directory ensured: {self.persist_directory}")
        except Exception as e:
            logger.error(f"Error creating persist directory: {e}")
            raise e
        
    def initialize_vector_store(self, embedding_function: Optional[Embeddings] = None):
        if embedding_function:
            self.embedding_function = embedding_function
        
        if not self.embedding_function:
            raise ValueError("Embedding function must be provided")
        
        try:
            logger.info("Initializing vector store")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise e
        
    def add_documents(self, documents: List[Document]) -> bool:
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
            
            logger.info(f"Adding {len(documents)} document(s) to vector store")
            self.vector_store.add_documents(documents)
            logger.info("Documents added successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            return False
        
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
            
            logger.info(f"Performing similarity search for query: '{query[:50]}...'")
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            return []
        
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
            
            logger.info(f"Performing similarity search with scores for query: '{query[:50]}...'")
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(results)} similar documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search with scores: {e}")
            return []
        
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
            
            default_kwargs = {"k": 5}
            if search_kwargs:
                default_kwargs.update(search_kwargs)
            
            retriever = self.vector_store.as_retriever(search_kwargs=default_kwargs)
            logger.info("Retriever created successfully")
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            raise e
        
    def get_collection_stats(self) -> dict:
        try:
            if not self.vector_store:
                return {'total_documents': 0, 'collection_name': None}
            
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                'total_documents': count,
                'collection_name': collection.name,
                'persist_directory': self.persist_directory
            }
        
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'total_documents': 0, 'collection_name': None}
        
    def clear_vector_store(self) -> bool:
        try:
            if not self.vector_store:
                return True
            
            logger.info("Clearing vector store")
            self.vector_store._collection.delete(where={})
            logger.info("Vector store cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False
        
    def is_initialized(self) -> bool:
        return self.vector_store is not None 


        

        
    
            

