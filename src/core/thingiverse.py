import requests
import time
import logging
from typing import List, Dict, Optional
from pathlib import Path
from config.settings import settings

logger = logging.getLogger(__name__)

class ThingiverseClient:
    """
    Client for interacting with Thingiverse API to fetch 3D parts.
    """
    
    def __init__(self):
        self.token = settings.THINGIVERSE_TOKEN
        self.base_url = settings.THINGIVERSE_API_BASE
        
        if not self.token:
            logger.warning("THINGIVERSE_TOKEN is missing. Scraper will fail unless using mocked data.")

        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

    def search_things(self, term: str, limit: int = 10) -> List[Dict]:
        """Search for things by term."""
        endpoint = f"{self.base_url}/search/{term}"
        params = {"per_page": limit, "sort": "relevant"}
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            # The API structure varies, usually returns a list or 'hits'
            data = response.json()
            return data if isinstance(data, list) else data.get('hits', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Search failed for '{term}': {e}")
            return []

    def get_thing_files(self, thing_id: int) -> List[Dict]:
        """Get list of files for a specific thing."""
        endpoint = f"{self.base_url}/things/{thing_id}/files"
        try:
            response = requests.get(endpoint, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get files for thing {thing_id}: {e}")
            return []

    def download_asset(self, url: str, dest_path: Path) -> bool:
        """Download a generic asset (STL or Image)."""
        if dest_path.exists():
            logger.info(f"Skipping {dest_path.name}, already exists.")
            return True
            
        try:
            # Thingiverse download links often redirect
            response = requests.get(url, allow_redirects=True, stream=True)
            response.raise_for_status()
            
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Simple rate limiting
            time.sleep(0.5) 
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Download failed for {url}: {e}")
            return False