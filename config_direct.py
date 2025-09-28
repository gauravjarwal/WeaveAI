"""
WeaveAI Direct Configuration
Alternative to .env file - edit the values below directly
"""

class Config:
    # ========================================
    # EDIT THESE VALUES WITH YOUR SETTINGS
    # ========================================
    
    # OpenAI API Configuration (REQUIRED)
    OPENAI_API_BASE = "https://your-azure-openai-endpoint.openai.azure.com"  # Replace with your endpoint
    OPENAI_API_KEY = "your-api-key-here"  # Replace with your API key
    OPENAI_API_VERSION = "2023-12-01-preview"
    OPENAI_DEPLOYMENT_NAME = "gpt-4"  # Replace with your deployment name
    
    # Vector Database Configuration
    CHROMA_PERSIST_DIRECTORY = "./chroma_db"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Document Processing
    MAX_FILE_SIZE_MB = 50
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # RAG Configuration
    TOP_K_DOCUMENTS = 5
    CONFIDENCE_THRESHOLD = 0.7
    
    # Server Configuration
    HOST = "0.0.0.0"
    PORT = 8000
    
    # Upload Configuration
    UPLOAD_DIRECTORY = "./uploads"
    ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

config = Config()

# ========================================
# USAGE INSTRUCTIONS
# ========================================
# 1. Edit the values above with your actual configuration
# 2. Rename this file to 'config.py' (replace the existing one)
# 3. Run: python main.py 