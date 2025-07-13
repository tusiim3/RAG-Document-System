# RAG Document System

A simple document interaction system using Retrieval-Augmented Generation (RAG) with Streamlit and Google's Gemini AI.

## Features

- Upload text documents (.txt files)
- Ask questions about your documents
- Get AI-powered answers with source citations
- Persistent vector database storage
- Clean web interface

## Setup

### With Docker (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/tusiim3/RAG-Document-System.git
cd RAG-Document-System
```

2. Copy `.env.example` to `.env` and add your Google API key:
```bash
cp .env.example .env
```

3. Run with Docker Compose:
```bash
docker-compose up --build
```

4. Open http://localhost:8501 in your browser

### Without Docker

1. Clone the repository:
```bash
git clone https://github.com/tusiim3/RAG-Document-System.git
cd RAG-Document-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your Google API key:
```bash
cp .env.example .env
```

4. Run the application:
```bash
streamlit run app.py
```

5. Open http://localhost:8501 in your browser

## Environment Variables

Required in `.env` file:
- `GOOGLE_API_KEY` - Your Google API key for Gemini
- `CHUNK_SIZE` - Text chunk size (default: 1000)
- `CHUNK_OVERLAP` - Chunk overlap (default: 200)
- `EMBEDDING_MODEL` - Embedding model name
- `LLM_TEMPERATURE` - AI response temperature (default: 0.3)

## Usage

1. Upload a text document using the file uploader
2. Wait for document processing to complete
3. Ask questions about the document in the chat interface
4. View source documents for each answer

## Technology Stack

- Streamlit for web interface
- LangChain for document processing
- ChromaDB for vector storage
- Google Gemini for AI responses
- Docker for containerization

##
