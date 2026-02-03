import sys
import json
import logging
from pathlib import Path
from tqdm import tqdm # Progress bar

# Add project root to pythonpath
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from src.core.embedding.clip_model import ClipEmbedder
from src.core.database.vector_store import PineconeDB

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger("transformers").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

def load_ledger():
    """Load the list of IDs that have already been indexed."""
    ledger_path = settings.PROCESSED_DATA_DIR / "indexed_ids.json"
    if ledger_path.exists():
        with open(ledger_path, 'r') as f:
            return set(json.load(f))
    return set()

def save_ledger(indexed_ids):
    """Save the updated list of indexed IDs."""
    ledger_path = settings.PROCESSED_DATA_DIR / "indexed_ids.json"
    with open(ledger_path, 'w') as f:
        json.dump(list(indexed_ids), f)

def main():
    logger.info("Starting Phase 2: AI Indexing (Smart Mode)...")

    # 1. Load Metadata (The Source)
    metadata_path = settings.PROCESSED_DATA_DIR / "metadata.json"
    if not metadata_path.exists():
        logger.error("metadata.json not found. Run Phase 1 first.")
        return

    with open(metadata_path, 'r') as f:
        all_records = json.load(f)
    
    # 2. Load Ledger (The Memory)
    indexed_ids = load_ledger()
    logger.info(f"Total parts in library: {len(all_records)}")
    logger.info(f"Parts already indexed:  {len(indexed_ids)}")

    # 3. Calculate the Delta (New items only)
    new_records = [r for r in all_records if str(r['id']) not in indexed_ids]
    
    if not new_records:
        logger.info("✅ No new parts to index. Database is up to date.")
        return

    logger.info(f"🚀 New parts to process: {len(new_records)}")

    # 4. Initialize AI Components (Only if we have work to do)
    try:
        embedder = ClipEmbedder()
        vector_db = PineconeDB()
    except Exception as e:
        logger.critical("Failed to initialize AI components. Check API Keys.")
        return

    # 5. Process New Data
    batch_size = 50
    batch = []
    processed_count = 0
    
    for record in tqdm(new_records, desc="Embedding New Images"):
        try:
            image_path = settings.BASE_DIR / record['thumbnail_path']
            
            if not image_path.exists():
                logger.warning(f"Image missing: {image_path}, skipping.")
                continue

            # Generate Embedding
            vector = embedder.get_image_embedding(image_path)
            
            if not vector:
                continue

            pinecone_vector = {
                "id": str(record['id']),
                "values": vector,
                "metadata": {
                    "name": record['name'],
                    "category": record.get('category', 'unknown'),
                    "width_mm": float(record['dims_mm']['x']),
                    "depth_mm": float(record['dims_mm']['y']),
                    "height_mm": float(record['dims_mm']['z']),
                    "stl_path": record['stl_path'],
                    "thumbnail_path": record['thumbnail_path']
                }
            }
            batch.append(pinecone_vector)

            # Upload Batch
            if len(batch) >= batch_size:
                vector_db.upsert_vectors(batch)
                # Update our local memory of what's done
                for item in batch:
                    indexed_ids.add(item['id'])
                save_ledger(indexed_ids)
                batch = []
                
        except Exception as e:
            logger.error(f"Error processing record {record['id']}: {e}")

    # Upload remaining items
    if batch:
        vector_db.upsert_vectors(batch)
        for item in batch:
            indexed_ids.add(item['id'])
        save_ledger(indexed_ids)

    logger.info("Phase 2 Complete. Ledger updated.")

if __name__ == "__main__":
    main()