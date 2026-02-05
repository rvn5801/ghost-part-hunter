import os
from google.cloud import storage
from config.settings import settings

class GoogleCloudStorage:
    def __init__(self):
        self.credentials_path = "service-account.json"
        # UPDATE THIS WITH YOUR ACTUAL BUCKET NAME
        self.bucket_name = "ghost-part-hunter-nani" 
        
        if os.path.exists(self.credentials_path):
            self.client = storage.Client.from_service_account_json(self.credentials_path)
            self.bucket = self.client.bucket(self.bucket_name)
        else:
            print("⚠️ Cloud Key missing! Uploads will fail.")
            self.client = None

    def upload_file(self, file_bytes, destination_blob_name: str) -> str:
        """Uploads bytes directly to GCS and returns public URL."""
        if not self.client: return ""

        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_string(file_bytes, content_type="image/jpeg")
        
        # Make public for the frontend
        try:
            blob.make_public()
        except:
            pass # ACLs might be restricted, but try anyway
            
        return blob.public_url