import sys
import time
import random
from typing import Any
from pathlib import Path
from pinecone import Pinecone

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import settings

def main():
    print(f"🕵️ STARTING DEEP DIAGNOSTIC...")
    
    try:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        
        # 1. Check Stats & Namespaces
        stats = index.describe_index_stats()
        print("\n📊 INDEX STATS:")
        print(stats)
        
        # Check if we have namespaces
        # We assume 'stats' is an object; accessing namespaces safely
        namespaces = getattr(stats, 'namespaces', {})
        # If it's a dict (older SDK), try get
        if isinstance(stats, dict):
            namespaces = stats.get('namespaces', {})

        if not namespaces:
            print("⚠️ WARNING: No namespaces found. Are vectors truly indexed?")
        else:
            print(f"✅ Found Namespaces: {list(namespaces.keys())}")

        # 2. Try to Fetch a Known ID
        raw_dir = settings.RAW_DATA_DIR
        # Find all numeric folders
        ids = [d.name for d in raw_dir.glob("*/*") if d.is_dir() and d.name.isdigit()]
        
        if not ids:
            print("❌ Could not find any local IDs to test with.")
            return

        test_id = ids[0]
        print(f"\n🧪 TESTING FETCH for ID: {test_id}")
        
        # Try fetching from default namespace
        fetch_result = index.fetch(ids=[test_id])
        
        # Safe Access logic
        vectors = getattr(fetch_result, 'vectors', {})
        
        if vectors:
            print(f"✅ SUCCESS: Found ID {test_id} in Default Namespace.")
        else:
            print(f"❌ FAILED: ID {test_id} not found in Default Namespace.")
            
            for ns in namespaces:
                print(f"   -> Trying namespace '{ns}'...")
                fetch_result_ns = index.fetch(ids=[test_id], namespace=ns)
                vectors_ns = getattr(fetch_result_ns, 'vectors', {})
                
                if vectors_ns:
                    print(f"   ✅ FOUND in namespace '{ns}'!")
                else:
                    print(f"   ❌ Not found in '{ns}'.")

        # 3. Test a Dummy Query
        print("\n🧪 TESTING DUMMY QUERY (Random Vector)...")
        dummy_vector = [random.random() for _ in range(512)]
        
        # Query without namespace
        raw_q_res = index.query(vector=dummy_vector, top_k=5)
        # Cast to Any to bypass Pylance strict checks
        q_res: Any = raw_q_res
        
        # Safe access to matches
        matches = getattr(q_res, 'matches', [])
        print(f"   -> Default Namespace Matches: {len(matches)}")
        
        # Query with all found namespaces
        for ns in namespaces:
            raw_q_res_ns = index.query(vector=dummy_vector, top_k=5, namespace=ns)
            q_res_ns: Any = raw_q_res_ns
            matches_ns = getattr(q_res_ns, 'matches', [])
            print(f"   -> Namespace '{ns}' Matches: {len(matches_ns)}")
                
    except Exception as e:
        print(f"❌ Fatal Error: {e}")

if __name__ == "__main__":
    main()