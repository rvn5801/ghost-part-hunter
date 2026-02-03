import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import logging
from typing import List, Union, Any
from pathlib import Path
from config.settings import settings

logger = logging.getLogger(__name__)

class ClipEmbedder:
    """
    Wrapper for OpenAI's CLIP model to generate image embeddings.
    """
    def __init__(self):
        logger.info(f"Loading CLIP Model: {settings.EMBEDDING_MODEL}...")
        try:
            # We use 'type: ignore' here because Pylance struggles with Transformers' dynamic instantiation
            self.model: CLIPModel = CLIPModel.from_pretrained(settings.EMBEDDING_MODEL) # type: ignore
            self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(settings.EMBEDDING_MODEL) # type: ignore
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Pylance sometimes flags string devices ("cpu") as invalid for .to(), so we ignore it
            self.model = self.model.to(self.device) # type: ignore
            
            logger.info(f"Model loaded on {self.device}")
        except Exception as e:
            logger.critical(f"Failed to load CLIP model: {e}")
            raise e

    def get_image_embedding(self, image_source: Union[str, Path, Image.Image]) -> List[float]:
        """
        Generate a vector embedding for a single image.
        """
        try:
            # Load Image if it's a path
            if isinstance(image_source, (str, Path)):
                image = Image.open(image_source)
            else:
                image = image_source

            # Preprocess and move to device
            # 'return_tensors' is a dynamic argument in Transformers, so Pylance can't see it staticially.
            # We suppress the error with 'type: ignore'
            inputs = self.processor(images=image, return_tensors="pt") # type: ignore
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inference (No Gradient calculation needed)
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)

            # Normalize the embedding (Critical for Cosine Similarity)
            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            
            # Convert to standard list of floats
            return outputs.cpu().numpy().tolist()[0]

        except Exception as e:
            logger.error(f"Error embedding image: {e}")
            return []