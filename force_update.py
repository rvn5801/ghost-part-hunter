import os

# 1. Define the path to the stubborn file
target_path = os.path.join("src", "core", "embedding", "clip_model.py")

# 2. Define the CORRECT code (The V2.0 Fix)
new_code = """
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import logging
from typing import cast, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClipEmbedder:
    def __init__(self):
        # --- PROOF OF LIFE PRINT ---
        print("\\n\\n🔥🔥🔥 V2.0 CODE LOADED SUCCESSFULLY - ZOMBIE KILLED 🔥🔥🔥\\n\\n") 
        
        logger.info("Loading CLIP model...")
        model_id = "openai/clip-vit-base-patch32"
        
        try:
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model = CLIPModel.from_pretrained(model_id)
            self.model.eval()
            logger.info("✅ CLIP model loaded successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to load CLIP model: {e}")
            raise e

    def get_image_embedding(self, image: Image.Image):
        try:
            # type: ignore suppresses VS Code errors
            inputs = self.processor(images=image, return_tensors="pt", padding=True) # type: ignore
            
            with torch.no_grad():
                # CRITICAL FIX: Explicitly get features as Tensor
                image_features = self.model.get_image_features(**inputs)
                
                # Force cast to Tensor for safety
                tensor_features = cast(torch.Tensor, image_features)
                
                # Normalize
                norm = tensor_features.norm(p=2, dim=-1, keepdim=True)
                normalized_features = tensor_features / norm
                
                return normalized_features.squeeze().tolist()
                
        except Exception as e:
            logger.error(f"Error embedding image: {e}")
            return [0.0] * 512 

    def get_text_embedding(self, text: str):
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True) # type: ignore
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                tensor_features = cast(torch.Tensor, text_features)
                
                norm = tensor_features.norm(p=2, dim=-1, keepdim=True)
                normalized_features = tensor_features / norm
                
                return normalized_features.squeeze().tolist()
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return [0.0] * 512
"""

# 3. Force Write the File
print(f"Overwriting {target_path}...")
with open(target_path, "w", encoding="utf-8") as f:
    f.write(new_code)

print("✅ SUCCESS! The file has been forcibly updated.")
print("Now run 'python -m src.backend.main'")