import sys
import os
from pathlib import Path
from pinecone import Pinecone

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import settings

def main():
    print(f"Connecting to Pinecone Index: {settings.PINECONE_INDEX_NAME}...")
    
    try:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        
        # Get Stats
        stats = index.describe_index_stats()
        
        print("\n📊 DATABASE STATUS:")
        print(f"Total Vectors: {stats.total_vector_count}")
        
        if stats.total_vector_count == 0:
            print("❌ FAILURE: The database is EMPTY.")
            print("   -> The Phase 2 script likely skipped your data.")
        else:
            print("✅ SUCCESS: Data exists.")
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    main()