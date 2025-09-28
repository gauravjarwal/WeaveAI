import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # OpenAI API Configuration
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "https://your-azure-openai-endpoint.openai.azure.com")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    OPENAI_API_VERSION: str = os.getenv("OPENAI_API_VERSION", "2023-12-01-preview")
    OPENAI_DEPLOYMENT_NAME: str = os.getenv("OPENAI_DEPLOYMENT_NAME", "gpt-4")
    
    # Vector Database Configuration
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Document Processing
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # RAG Configuration
    TOP_K_DOCUMENTS: int = int(os.getenv("TOP_K_DOCUMENTS", "5"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Upload Configuration
    UPLOAD_DIRECTORY: str = os.getenv("UPLOAD_DIRECTORY", "./uploads")
    ALLOWED_EXTENSIONS: set = {".pdf", ".docx", ".txt", ".md"}

config = Config() 