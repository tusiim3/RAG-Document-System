import logging
from typing import List, Optional
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self, model_name: str = "text-embedding-3-small", device: str = 'cpu'):
        self.model_name = model_name
        self.device = device
        self.embeddings = None
        self._initialize_embeddings()
        
    def _initialize_embeddings(self):
        try:
            logger.info(f"Initializing embedding model: {self.model_name}")
            self.embeddings = OpenAIEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': self.device}  # whats kwargs?   whats async?
            )
            logger.info("Embedding model initialized successfully") 

        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise e
    
    def get_embeddings(self) -> OpenAIEmbeddings:
        if self.embeddings is None:
            self._initialize_embeddings()
        return self.embeddings
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            logger.info(f"Generating embeddings for {len(texts)} text(s)")
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise e
        
    def generate_single_embedding(Self, text: str) -> List[float]:
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            raise e
        
    def get_embedding_dimension(self) -> int:
        try:
            test_embedding = self.generate_single_embedding("test")
            return len(test_embedding)
        
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {e}")
            raise e
        
    def get_model_info(self) -> dict:
        return {
            'model_name': self.model_name,
            'device': self.device,
            'dimension': self.get_embedding_dimension(),
            'is_initialized': self.embeddings is not None
        } 

            