import json
import sys
from pathlib import Path
import random

# Add project root to pythonpath
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import settings

def main():
    print("🔍 Starting Phase 1 Verification...\n")
    
    metadata_path = settings.PROCESSED_DATA_DIR / "metadata.json"
    
    # 1. Check Metadata File Existence
    if not metadata_path.exists():
        print("❌ CRITICAL FAIL: metadata.json not found.")
        print("   -> Run 'python scripts/build_dataset.py' first.")
        return

    try:
        with open(metadata_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("❌ CRITICAL FAIL: metadata.json is corrupted (invalid JSON).")
        return

    total_records = len(data)
    print(f"✅ Metadata loaded. Total Records: {total_records}")

    if total_records == 0:
        print("❌ CRITICAL FAIL: Metadata list is empty. Scraper found nothing.")
        return

    # 2. Audit Data Integrity
    missing_files = 0
    zero_dimensions = 0
    valid_records = 0
    
    print("\nauditing records...")
    
    for item in data:
        # Check Paths
        stl_path = settings.BASE_DIR / item['stl_path']
        thumb_path = settings.BASE_DIR / item['thumbnail_path']
        
        file_missing = False
        if not stl_path.exists():
            print(f"   ⚠️ Missing STL: {item['id']}")
            file_missing = True
        if not thumb_path.exists():
            print(f"   ⚠️ Missing Thumb: {item['id']}")
            file_missing = True
            
        if file_missing:
            missing_files += 1
            continue

        # Check Geometry
        dims = item['dims_mm']
        volume = dims['x'] * dims['y'] * dims['z']
        
        if volume <= 0:
            print(f"   ⚠️ Zero Volume detected for ID {item['id']}")
            zero_dimensions += 1
            continue
            
        valid_records += 1

    # 3. Final Report
    print("-" * 30)
    print("📊 VERIFICATION REPORT")
    print("-" * 30)
    print(f"Total Entries:      {total_records}")
    print(f"Missing Files:      {missing_files}")
    print(f"Bad Geometry:       {zero_dimensions}")
    print(f"✅ VALID RECORDS:   {valid_records}")
    print("-" * 30)

    # 4. Success Logic
    if valid_records >= 5:
        print("\n🚀 PHASE 1 SUCCESS: Pipeline is functional.")
        print("   You have enough valid data to verify the AI in Phase 2.")
    else:
        print("\n❌ PHASE 1 FAILED: Not enough valid data.")
        print("   Check your internet connection, API Token, or scrape logic.")

if __name__ == "__main__":
    main()