import logging
import os
from typing import List, Optional, Tuple
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
import google.generativeai as genai

from .document_processor import DocumentProcessor
from .embedding_manager import EmbeddingManager
from .vector_store import VectorStoreManager

load_dotenv()

logger = logging.getLogger(__name__)

# Load API key from .env file
google_api_key = os.environ.get("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

class RAGPipeline:
    def __init__(self, api_key: Optional[str] = None, chunk_size: int = 1000, chunk_overlap: int = 200, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", persist_directory: str = "./chroma_db", temperature: float = 0.3):
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

            genai.configure(api_key=google_api_key)

            self.document_processor = DocumentProcessor(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            self.embedding_manager = EmbeddingManager(model_name=self.embedding_model)
            self.vector_store_manager = VectorStoreManager(persist_directory=self.persist_directory, embedding_function=self.embedding_manager.get_embeddings())
            self.vector_store_manager.initialize_vector_store()
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=self.temperature)

            logger.info("RAG Pipeline components initialized successfully") 

        except Exception as e:
            logger.error(f"Error initializing RAG Pipeline components: {e}")
            raise e

    def process_document(self, file_path: str) -> bool: 
        try:
            logger.info(f"Processing document: {file_path}")
            # Chunk document
            chunks = self.document_processor.process_document(file_path)
            if not chunks:
                logger.error("No chunks generated from document")
                return False
            # Add chunks to vector store
            success = self.vector_store_manager.add_documents(chunks)
            if not success:
                logger.error("Failed to add chunks to vector store")
                return False
            # Initialize QA chain
            retriever = self.vector_store_manager.get_retriever()
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )

            logger.info(f"Document processed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return False
        
    def query(self, question: str) -> Tuple[str, List[Document]]:
        try:
            if not self.qa_chain:
                return "Please process a document first before asking questions.", []
            logger.info(f"Processing query: '{question}'")
            response = self.qa_chain({"query": question})
            answer = response['result']
            source_docs = response.get("source_documents", [])
            logger.info(f"Query completed successfully. Answer length: {len(answer)}")
            return answer, source_docs
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error processing query: {str(e)}", []
        
    def get_system_info(self) -> dict:
        try:
            info = {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'embedding_model': self.embedding_model,
                'persist_directory': self.persist_directory,
                'temperature': self.temperature,
                'components_initialized': {
                    'document_processor': self.document_processor is not None,
                    'embedding_manager': self.embedding_manager is not None,
                    'vector_store_manager': self.vector_store_manager is not None,
                    'llm': self.llm is not None,
                    'qa_chain': self.qa_chain is not None
                }
            }

             # Add embedding model info
            if self.embedding_manager:
                info['embedding_info'] = self.embedding_manager.get_model_info()
            
            # Add vector store stats
            if self.vector_store_manager:
                info['vector_store_stats'] = self.vector_store_manager.get_collection_stats()
            
            return info
        
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {}
        
    def clear_knowledge_base(self) -> bool:
         try:
            logger.info("Clearing knowledge base")
            
            # Clear vector store
            if self.vector_store_manager:
                self.vector_store_manager.clear_vector_store()
            
            # Reset QA chain
            self.qa_chain = None
            
            logger.info("Knowledge base cleared successfully")
            return True
         
         except Exception as e:
            logger.error(f"Error clearing knowledge base: {e}")
            return False
         
    def is_ready(self) -> bool:
        return (
            self.document_processor is not None and
            self.embedding_manager is not None and
            self.vector_store_manager is not None and
            self.llm is not None and
            self.qa_chain is not None
        )


            
            

