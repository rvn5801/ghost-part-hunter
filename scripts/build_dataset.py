import sys
import os
import json
import logging
from pathlib import Path

# Add project root to pythonpath
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from src.core.thingiverse import ThingiverseClient
from src.core.geometry import GeometryEngine

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.DATA_DIR / "logs" / "scraper.log"),
        logging.StreamHandler()
    ]
)
# Suppress the noisy numpy-stl warnings
logging.getLogger('numpy_stl').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Dataset Build Phase (Organized Mode)...")
    
    client = ThingiverseClient()
    
    if not client.token:
        logger.error("No API Token found. Please check .env file.")
        sys.exit(1)

    metadata_registry = []
    
    # 1. Iterate over search terms (Categories)
    for term in settings.SEARCH_TERMS:
        logger.info(f"--- Processing Category: {term.upper()} ---")
        
        # Create category directory: data/raw/gear, data/raw/bracket, etc.
        category_dir = settings.RAW_DATA_DIR / term
        category_dir.mkdir(parents=True, exist_ok=True)
        
        things = client.search_things(term, limit=10) # Keeping limit low for dev
        
        for thing in things:
            thing_id = thing.get('id')
            thing_name = thing.get('name', 'Unknown')
            thumbnail_url = thing.get('thumbnail')
            
            if not thing_id:
                continue

            # Create a dedicated directory for this Thing inside the category folder
            # Structure: data/raw/{term}/{id}/
            thing_dir = category_dir / str(thing_id)
            
            # Optimization: Skip if we already scraped this ID successfully
            if thing_dir.exists() and any(thing_dir.glob("*.stl")):
                logger.info(f"Skipping {thing_name} (Already exists)")
                continue

            thing_dir.mkdir(exist_ok=True)
            logger.info(f"Processing: {thing_name} ({thing_id})")
            
            # 2. Download Thumbnail
            thumb_path = thing_dir / "thumbnail.jpg"
            if thumbnail_url:
                client.download_asset(thumbnail_url, thumb_path)
            
            # 3. Get Files and Find STL
            files = client.get_thing_files(thing_id)
            stl_found = False
            
            for file_info in files:
                filename = file_info.get('name', '').lower()
                download_url = file_info.get('download_url')
                
                # We only want STLs
                if filename.endswith('.stl') and download_url:
                    stl_path = thing_dir / file_info['name']
                    
                    # Download
                    success = client.download_asset(download_url, stl_path)
                    
                    if success:
                        # 4. Analyze Geometry
                        dims = GeometryEngine.get_bounding_box(stl_path)
                        
                        if dims:
                            width, depth, height = dims
                            # Log dimensions cleanly
                            logger.info(f"   -> [Size]: {width:.1f} x {depth:.1f} x {height:.1f} mm")
                            
                            # Add to registry
                            record = {
                                "id": thing_id,
                                "name": thing_name,
                                "category": term,  # Added Category field
                                "thumbnail_path": str(thumb_path.relative_to(settings.BASE_DIR)),
                                "stl_path": str(stl_path.relative_to(settings.BASE_DIR)),
                                "dims_mm": {
                                    "x": width,
                                    "y": depth,
                                    "z": height
                                },
                                "volume": GeometryEngine.get_volume(stl_path)
                            }
                            metadata_registry.append(record)
                            stl_found = True
                            break # Found one valid STL, move to next Thing
            
            if not stl_found:
                # Cleanup empty directory if no STL found
                try:
                    os.rmdir(thing_dir)
                except:
                    pass

    # 5. Save Metadata Registry
    output_file = settings.PROCESSED_DATA_DIR / "metadata.json"
    with open(output_file, 'w') as f:
        json.dump(metadata_registry, f, indent=2)
    
    logger.info(f"Phase 1 Complete. Metadata saved to {output_file}. Total records: {len(metadata_registry)}")

if __name__ == "__main__":
    main()