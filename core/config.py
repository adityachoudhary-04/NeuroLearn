import os
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "PDF RAG & Q/A Learning Assistant"
    DEBUG: bool = True
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    CHROMA_PERSIST_DIR: Path = BASE_DIR / "data" / "chroma_db"
    
    # LLM Providers: "gemini", "groq", "openai", "claude", "mock"
    DEFAULT_LLM_PROVIDER: str = "gemini"
    DEFAULT_LLM_MODEL: str = "gemini-3.1-flash-lite"
    
    # Embedding Providers: "bert", "gemini", "openai", "local"
    DEFAULT_EMBEDDING_PROVIDER: str = "bert"
    DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Content Selection / Salience Methods: "tfidf", "textrank", "embedding"
    DEFAULT_SALIENCE_METHOD: str = "tfidf"
    
    # API Keys
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    
    # Chunking Configuration
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    TOP_K_RETRIEVAL: int = 5
    
    class Config:
        env_file = BASE_DIR / ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
