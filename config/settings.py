# from pydantic_settings import BaseSettings

# class Settings(BaseSettings):
#     APP_NAME: str = 'Ghost Part Hunter'
#     class Config:


import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"

    # Thingiverse API
    THINGIVERSE_TOKEN = os.getenv("THINGIVERSE_TOKEN")
    THINGIVERSE_API_BASE = "https://api.thingiverse.com"
    
    # Scraper Settings
    MAX_ITEMS_TO_FETCH = 50  # Limit for MVP/Testing
    SEARCH_TERMS = ["gear", "bracket", "knob", "clip", "lever"]

    # AI / Pinecone Settings
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = "us-east-1" # Default for free tier, usually ignored in serverless
    PINECONE_INDEX_NAME = "ghost-part-hunter"
    EMBEDDING_MODEL = "openai/clip-vit-base-patch32"
    VECTOR_DIMENSION = 512

    def ensure_dirs(self):
        """Ensure critical data directories exist."""
        self.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_dirs()