import logging
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.schema import Document

from .config import Config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
       config = Config.get_doc_processing_config()
       self.chunk_size = chunk_size  or config['chunk_size']
       self.chunk_overlap = chunk_overlap or config['chunk_overlap']
       self.text_splitter = RecursiveCharacterTextSplitter(
           chunk_size=self.chunk_size,
           chunk_overlap=self.chunk_overlap,
           length_fucntion=len,
           separators=["\n\n", "\n", " ", ""]
         )

    def load_document(self, file_path: str,  encoding: Optional[str] = None) -> List[Document]:
        try:
            config = Config.get_doc_processing_config()
            encoding = encoding or config['encoding']
            logger.info(f"Loading document from {file_path}")
            loader = TextLoader(file_path, encoding=encoding)
            documents = loader.load()
            logger.info(f"Successfully loaded {len(documents)} document(s)")
            return documents
        
        except Exception as e:
            logger.error(f"Error loading document from {file_path}: {e}")
            raise e
        
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        try:
            logger.info(f"Chunking {len(documents)} document(s)")
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Successfully created {len(chunks)} chunk(s)")
            return chunks
        
        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            raise e
        
    def process_document(self, file_path: str) -> List[Document]:
        try:
            documents = self.load_document(file_path)
            chunks = self.chunk_documents(documents)
            logger.info(f"Document processing completed: {len(chunks)} chunks created")
            return chunks
        
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise e
        
    def get_document_stats(self, chunks: List[Document]) -> dict:
        if not chunks:
            return {
                'total_chunks': 0,
                'total_characters': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0
            }
        
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        total_chars = sum(chunk_sizes)

        return {
            'total_chunks': len(chunks),
            'total_characters': total_chars,
            'avg_chunk_size': total_chars / len(chunks),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes)
        }

        