# config.py - Configuration Management
import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Configuration
    API_TITLE: str = "Strategy AI Backend"
    API_VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_ORIGINS: List[str] = ["*"]  # Configure for production
    
    # Database Configuration
    SUPABASE_URL: str
    SUPABASE_SERVICE_KEY: str
    
    # Vector Store Configuration
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "strategy-docs"
    PINECONE_ENVIRONMENT: str = "us-east-1"
    
    # AI Services Configuration
    OPENAI_API_KEY: str
    AI_MODEL_NAME: str = "gpt-4o-mini"
    EMBEDDING_MODEL_NAME: str = "text-embedding-ada-002"
    AI_TEMPERATURE: float = 0.2
    
    # Admin Configuration
    ADMIN_API_KEY: str = "admin-secret-key-change-in-production"
    
    # File Processing Configuration
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_FILE_TYPES: List[str] = [".pdf", ".docx", ".txt", ".csv", ".md"]
    TEMP_UPLOAD_DIR: str = "/tmp"
    
    # Document Processing Configuration
    DEFAULT_CHUNK_SIZE: int = 1000
    DEFAULT_CHUNK_OVERLAP: int = 200
    MAX_CHUNKS_PER_DOCUMENT: int = 1000
    
    # Vector Search Configuration
    VECTOR_SEARCH_TOP_K: int = 8
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Web Scraping Configuration
    MAX_SCRAPING_PAGES: int = 50
    SCRAPING_TIMEOUT_SECONDS: int = 30
    SCRAPING_DELAY_MS: int = 1000
    
    # Background Tasks Configuration
    TASK_QUEUE_SIZE: int = 100
    TASK_TIMEOUT_MINUTES: int = 30
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Analytics Configuration
    ANALYTICS_RETENTION_DAYS: int = 365
    ENABLE_ANALYTICS: bool = True
    
    # Security Configuration
    API_RATE_LIMIT: str = "100/minute"
    CORS_MAX_AGE: int = 3600
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

# Environment-specific configurations
class DevelopmentSettings(Settings):
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:3001"]

class ProductionSettings(Settings):
    DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"
    # Set specific allowed origins in production
    ALLOWED_ORIGINS: List[str] = ["https://your-v0-app.vercel.app"]

class TestingSettings(Settings):
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    # Use test database and services
    SUPABASE_URL: str = "test_url"
    PINECONE_INDEX_NAME: str = "test-index"

def get_settings() -> Settings:
    """Get settings based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()

# Use environment-specific settings
settings = get_settings()
if settings.admin_secret_key == "admin-secret-key-change-in-production":
    raise ValueError("Please set a secure ADMIN_SECRET_KEY in production.")
