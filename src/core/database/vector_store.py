import time
import logging
from typing import List, Dict, Optional, Any
from pinecone import Pinecone, ServerlessSpec
from config.settings import settings

logger = logging.getLogger(__name__)

class PineconeDB:
    """
    Manager for Pinecone Vector Database interactions.
    """
    def __init__(self):
        if not settings.PINECONE_API_KEY:
            logger.critical("PINECONE_API_KEY is missing.")
            raise ValueError("No Pinecone API Key provided.")
            
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX_NAME
        self.index = None
        
        # Initialize Index Connection
        self._ensure_index_exists()
        
        # Explicitly initialize the index object
        self.index = self.pc.Index(self.index_name)

    def _ensure_index_exists(self):
        """Checks if index exists, creates it if not."""
        existing_indexes = [i.name for i in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            logger.info(f"Creating Pinecone Index: {self.index_name}")
            try:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=settings.VECTOR_DIMENSION, 
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                # Wait for index to initialize
                while True:
                    idx_desc = self.pc.describe_index(self.index_name)
                    # Check if status exists and is ready (handling object attribute access)
                    if idx_desc.status and idx_desc.status.ready:
                        break
                    time.sleep(1)
                
                logger.info("Index created and ready.")
            except Exception as e:
                logger.error(f"Failed to create index: {e}")
                raise e
        else:
            logger.info(f"Connected to existing index: {self.index_name}")

    def upsert_vectors(self, vectors: List[Any]):
        """
        Upload vectors to Pinecone.
        Format: [{'id': '1', 'values': [0.1...], 'metadata': {...}}, ...]
        """
        if self.index is None:
            logger.error("Index not initialized.")
            return

        try:
            # Upsert in batches (Pinecone SDK handles the typing validation internally)
            self.index.upsert(vectors=vectors)
            logger.info(f"Upserted {len(vectors)} vectors.")
        except Exception as e:
            logger.error(f"Upsert failed: {e}")

    def search(self, vector: List[float], top_k: int = 5, filter_dict: Optional[Dict[str, Any]] = None):
        """
        Query the database for similar vectors.
        """
        if self.index is None:
            logger.error("Index not initialized.")
            return {"matches": []}

        try:
            results = self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"matches": []}