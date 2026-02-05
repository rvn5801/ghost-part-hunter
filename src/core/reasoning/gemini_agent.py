import vertexai
from vertexai.generative_models import GenerativeModel, Part
import os
from dotenv import load_dotenv
import logging
from PIL import Image
import io

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env vars
load_dotenv()

class GeminiVerifier:
    def __init__(self):
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = "us-central1" # Or your specific region
        
        # Initialize Vertex AI
        try:
            vertexai.init(project=self.project_id, location=self.location)
            # Use 'gemini-1.5-flash-001' for speed and low cost
            self.model = GenerativeModel("gemini-2.5-flash-lite") 
            logger.info("✅ Gemini Agent Initialized.")
        except Exception as e:
            logger.error(f"❌ Failed to init Gemini: {e}")
            self.model = None

    def _process_image_input(self, img_input):
        """
        Helper: Converts PIL Images or direct bytes into raw JPEG bytes for Gemini.
        """
        # Case 1: Already bytes? Return them.
        if isinstance(img_input, bytes):
            return img_input
        
        # Case 2: PIL Image? Convert to bytes.
        if isinstance(img_input, Image.Image):
            buf = io.BytesIO()
            # Convert to RGB (removes transparency which JPEGs hate)
            if img_input.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img_input.size, (255, 255, 255))
                background.paste(img_input, mask=img_input.split()[3])
                img_input = background
            elif img_input.mode != 'RGB':
                img_input = img_input.convert('RGB')
            
            img_input.save(buf, format="JPEG")
            return buf.getvalue()
            
        # Case 3: Unknown? Fail gracefully.
        raise ValueError(f"Unsupported image format: {type(img_input)}")

    def analyze_match(self, user_img_data, candidate_img_data, part_name):
        """
        Compares user image with candidate image using Gemini.
        """
        if not self.model:
            return "Gemini not configured."

        try:
            # 1. Convert inputs to raw bytes (Fixes the TypeError)
            user_bytes = self._process_image_input(user_img_data)
            candidate_bytes = self._process_image_input(candidate_img_data)

            # 2. Create Vertex AI 'Parts'
            image1 = Part.from_data(user_bytes, mime_type="image/jpeg")
            image2 = Part.from_data(candidate_bytes, mime_type="image/jpeg")
            
            # 3. Construct the Prompt
            prompt = f"""
            You are a mechanical part verifier. 
            Image 1 is a photo of a part found by a user.
            Image 2 is the database reference image for a part named "{part_name}".
            
            Compare these two images. 
            1. Are they the same object? 
            2. Mention any key visual similarities or differences (shape, holes, color).
            3. Final Verdict: MATCH or NO MATCH.
            
            Keep your answer concise (max 3 sentences).
            """

            # 4. Generate Response
            response = self.model.generate_content(
                [image1, image2, prompt]
            )
            
            return response.text

        except Exception as e:
            logger.error(f"Gemini Analysis Failed: {e}")
            return f"Error analyzing match: {e}"