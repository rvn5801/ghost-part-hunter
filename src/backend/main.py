from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import uvicorn
import io
from PIL import Image

# Import Core Modules
from src.core.embedding.clip_model import ClipEmbedder
from src.core.database.vector_store import PineconeDB
from src.core.vision.coin_detector import CoinDetector

app = FastAPI(title="Ghost Part Hunter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("⚡ INITIALIZING AI MODELS...")
embedder = ClipEmbedder()
vector_db = PineconeDB()
print("✅ AI MODELS READY.")

class SearchResult(BaseModel):
    id: str
    name: str
    score: float
    thumbnail_url: str
    stl_path: str   # <--- NEW FIELD added here
    dimensions: Dict[str, Optional[float]]
    size_match: str 

@app.post("/search", response_model=List[SearchResult])
async def search_part(file: UploadFile = File(...)):
    print(f"\n--- 🔍 NEW SEARCH REQUEST: {file.filename} ---")
    
    content = await file.read()
    pixels_per_mm, processed_img = CoinDetector.get_scale_factor(content)

    try:
        pil_image = Image.open(io.BytesIO(content))
        query_vector = embedder.get_image_embedding(pil_image)
    except Exception as e:
        print(f"❌ ERROR in Embedding: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # Pylance Fix: Check connection
    if vector_db.index is None:
        raise HTTPException(status_code=500, detail="Database Connection Error")
    
    try:
        raw_results = vector_db.index.query(
            vector=query_vector, 
            top_k=20, 
            include_metadata=True,
            namespace="" 
        )
    except Exception as e:
        print(f"❌ ERROR in Pinecone Search: {e}")
        return []

    results_obj: Any = raw_results
    matches = []

    if isinstance(results_obj, dict):
        matches = results_obj.get('matches', [])
    elif hasattr(results_obj, 'matches'):
        matches = getattr(results_obj, 'matches', [])
    elif hasattr(results_obj, 'to_dict'):
        matches = results_obj.to_dict().get('matches', [])

    print(f"6. Matches found: {len(matches)}")

    output = []
    for match in matches:
        match_data: Any = match
        
        if hasattr(match_data, 'to_dict'):
            match_data = match_data.to_dict()
        elif not isinstance(match_data, dict):
             try: match_data = dict(match_data)
             except: continue

        meta = match_data.get('metadata', {})
        score = match_data.get('score', 0.0)
        
        output.append(SearchResult(
            id=str(match_data.get('id', 'unknown')),
            name=meta.get('name', 'Unknown'),
            score=score,
            thumbnail_url=meta.get('thumbnail_path', ''),
            stl_path=meta.get('stl_path', ''),  # <--- NEW: Extract STL path from metadata
            dimensions={
                "x": meta.get('width_mm'),
                "y": meta.get('depth_mm'),
                "z": meta.get('height_mm')
            },
            size_match="Visual Match" 
        ))

    print(f"✅ RETURNING {len(output)} RESULTS TO FRONTEND")
    return output

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Optional, Any, Dict
# import uvicorn
# import io
# from PIL import Image

# # Import Core Modules
# from src.core.embedding.clip_model import ClipEmbedder
# from src.core.database.vector_store import PineconeDB
# from src.core.vision.coin_detector import CoinDetector

# app = FastAPI(title="Ghost Part Hunter API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# print("⚡ INITIALIZING AI MODELS...")
# embedder = ClipEmbedder()
# vector_db = PineconeDB()
# print("✅ AI MODELS READY.")

# class SearchResult(BaseModel):
#     id: str
#     name: str
#     score: float
#     thumbnail_url: str
#     dimensions: Dict[str, Optional[float]]
#     size_match: str 

# @app.post("/search", response_model=List[SearchResult])
# async def search_part(file: UploadFile = File(...)):
#     print(f"\n--- 🔍 NEW SEARCH REQUEST: {file.filename} ---")
    
#     # 1. Read Image
#     content = await file.read()
    
#     # 2. Vision (Optional)
#     pixels_per_mm, processed_img = CoinDetector.get_scale_factor(content)

#     # 3. AI: Generate Embedding
#     try:
#         pil_image = Image.open(io.BytesIO(content))
#         query_vector = embedder.get_image_embedding(pil_image)
#     except Exception as e:
#         print(f"❌ ERROR in Embedding: {e}")
#         raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

#     # 4. Database: Search
#     print("4. Sending query to Pinecone...")
    
#     # --- PYLANCE FIX: Explicitly check for None ---
#     if vector_db.index is None:
#         print("❌ CRITICAL: Pinecone Index is not initialized.")
#         raise HTTPException(status_code=500, detail="Database Connection Error")
    
#     try:
#         # We explicitly target the default namespace (empty string)
#         # This maps to '__default__' in the Pinecone Stats
#         raw_results = vector_db.index.query(
#             vector=query_vector, 
#             top_k=20, 
#             include_metadata=True,
#             namespace=""  # <--- CRITICAL: Targets the default namespace
#         )
#     except Exception as e:
#         print(f"❌ ERROR in Pinecone Search: {e}")
#         return []

#     # 5. Parse Results (Safe Handling)
#     # We cast to 'Any' to disable strict Pylance checks for this block
#     results_obj: Any = raw_results
#     matches = []

#     # Robust extraction logic
#     if isinstance(results_obj, dict):
#         matches = results_obj.get('matches', [])
#     elif hasattr(results_obj, 'matches'):
#         matches = getattr(results_obj, 'matches', [])
#     elif hasattr(results_obj, 'to_dict'):
#         matches = results_obj.to_dict().get('matches', [])

#     print(f"6. Matches found: {len(matches)}")

#     output = []
#     for match in matches:
#         match_data: Any = match
        
#         # Safe attribute access
#         if hasattr(match_data, 'to_dict'):
#             match_data = match_data.to_dict()
#         elif not isinstance(match_data, dict):
#              # Try forcing dict if it's some other object
#              try: match_data = dict(match_data)
#              except: continue

#         meta = match_data.get('metadata', {})
#         score = match_data.get('score', 0.0)
        
#         output.append(SearchResult(
#             id=str(match_data.get('id', 'unknown')),
#             name=meta.get('name', 'Unknown'),
#             score=score,
#             thumbnail_url=meta.get('thumbnail_path', ''),
#             dimensions={
#                 "x": meta.get('width_mm'),
#                 "y": meta.get('depth_mm'),
#                 "z": meta.get('height_mm')
#             },
#             size_match="Visual Match" 
#         ))

#     print(f"✅ RETURNING {len(output)} RESULTS TO FRONTEND")
#     return output

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)