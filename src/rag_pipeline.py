import logging
import os
from typing import List, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

from .document_processor import DocumentProcessor
from .embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)

# Load API key from .env file
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

class RAGPipeline:
    def __init__(self, api_key: Optional[str] = None, chunk_size: int = 1000, chunk_overlap: int = 200, embedding_model: str = "text-embedding-3-small", persist_directory: str = "./chroma_db", temperature: float = 0.3):
        self.api_key = api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.temperature = temperature
        self.document_processor = None
        self.embedding_manager = None
        self.llm = None
        self.qa_chain = None

        self._initialize_components()

    def _initialize_components(self):
        try:
            logger.info("Initializing RAG Pipeline components")
            self.document_processor = DocumentProcessor(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            self.embedding_manager = EmbeddingManager(model_name=self.embedding_model)
            self.vector_store_manager = VectorStoreManager(persist_directory=self.persist_directory, embedding_function=self.embedding_manager.get_embeddings())
            self.llm = ChatOpenAI(temperature=self.temperature, api_key=self.api_key)

            logger.info("RAG Pipeline components initialized successfully") 

        except Exception as e:
            logger.error(f"Error initializing RAG Pipeline components: {e}")
            raise e

    def process_document(self, file_path: str) -> bool: 
        try:
            logger.info(f"Processing document: {file_path}")
            

