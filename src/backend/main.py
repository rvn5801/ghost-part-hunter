from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import uvicorn
import io
from PIL import Image
from pathlib import Path

# Import Core Modules
from src.core.embedding.clip_model import ClipEmbedder
from src.core.database.vector_store import PineconeDB
from src.core.vision.coin_detector import CoinDetector
from src.core.reasoning.gemini_agent import GeminiVerifier

app = FastAPI(title="Ghost Part Hunter API")

# Setup project root for finding local files
BASE_DIR = Path(__file__).resolve().parent.parent.parent

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("⚡ INITIALIZING AI MODELS...")
embedder = ClipEmbedder()
vector_db = PineconeDB()
gemini = GeminiVerifier()
print("✅ AI MODELS READY.")

class SearchResult(BaseModel):
    id: str
    name: str
    score: float
    thumbnail_url: str
    stl_path: str
    dimensions: Dict[str, Optional[float]]
    gemini_analysis: Optional[str] = "Pending"

@app.post("/search", response_model=List[SearchResult])
async def search_part(file: UploadFile = File(...)):
    print(f"\n--- 🔍 NEW SEARCH REQUEST: {file.filename} ---")
    
    content = await file.read()
    
    # 1. Load User Image (This is the variable we must use later)
    user_pil_image = Image.open(io.BytesIO(content))

    # AI: Generate Embedding
    try:
        query_vector = embedder.get_image_embedding(user_pil_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Check connection
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

    # Safe extraction logic
    results_obj: Any = raw_results
    matches = []
    if isinstance(results_obj, dict):
        matches = results_obj.get('matches', [])
    elif hasattr(results_obj, 'matches'):
        matches = getattr(results_obj, 'matches', [])
    elif hasattr(results_obj, 'to_dict'):
        matches = results_obj.to_dict().get('matches', [])

    output = []
    
    # --- GEMINI INTEGRATION ---
    for idx, match in enumerate(matches):
        match_data: Any = match
        if hasattr(match_data, 'to_dict'): match_data = match_data.to_dict()
        elif not isinstance(match_data, dict): match_data = dict(match_data)

        meta = match_data.get('metadata', {})
        score = match_data.get('score', 0.0)
        
        analysis_text = "Analysis skipped for speed"
        
        # Only analyze the #1 best match
        if idx == 0: 
            print("🤖 Asking Gemini to verify Match #1...")
            
            # Find the local candidate image
            thumb_rel_path = meta.get('thumbnail_path', '')
            candidate_path = BASE_DIR / thumb_rel_path
            
            # Fallback path logic
            if not candidate_path.exists():
                try:
                    parts = Path(thumb_rel_path).parts
                    if len(parts) > 3 and parts[1] == 'raw':
                        new_parts = list(parts)
                        new_parts.pop(2)
                        candidate_path = BASE_DIR / Path(*new_parts)
                except: pass
            
            if candidate_path.exists():
                candidate_pil = Image.open(candidate_path)
                
                # --- FIX: USE CORRECT VARIABLE NAMES ---
                analysis_text = gemini.analyze_match(
                    user_img_data=user_pil_image,         # Correct variable (Line 50)
                    candidate_img_data=candidate_pil,     # Correct variable (Line 114)
                    part_name=meta.get('name', 'Unknown') # Correct variable (Line 90)
                )
                print(f"🤖 Gemini says: {analysis_text}")
            else:
                analysis_text = "Candidate image not found locally."

        output.append(SearchResult(
            id=str(match_data.get('id', 'unknown')),
            name=meta.get('name', 'Unknown'),
            score=score,
            thumbnail_url=meta.get('thumbnail_path', ''),
            stl_path=meta.get('stl_path', ''),
            dimensions={
                "x": meta.get('width_mm'),
                "y": meta.get('depth_mm'),
                "z": meta.get('height_mm')
            },
            gemini_analysis=analysis_text 
        ))

    return output

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)