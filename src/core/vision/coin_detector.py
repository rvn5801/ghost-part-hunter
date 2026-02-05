import cv2
import numpy as np
from rembg import remove
from PIL import Image
import io
from typing import Union, cast, Any

class CoinDetector:
    def __init__(self):
        pass

    def _to_bytes(self, input_data: Any) -> bytes:
        """
        Helper to safely convert ANY input (Path, Bytes, Stream, PIL Image) into raw bytes.
        """
        # 1. If it's already bytes, return it
        if isinstance(input_data, bytes):
            return input_data
        
        # 2. If it's a bytearray, convert
        if isinstance(input_data, bytearray):
            return bytes(input_data)
        
        # 3. If it's a Numpy Array (OpenCV Image), encode it
        if isinstance(input_data, np.ndarray):
            success, encoded_img = cv2.imencode('.png', input_data)
            if success:
                return encoded_img.tobytes()
            return b""
        
        # 4. If it's a PIL Image, save it to a buffer
        if isinstance(input_data, Image.Image):
            buf = io.BytesIO()
            input_data.save(buf, format="PNG")
            return buf.getvalue()
        
        # 5. If it's a Stream/File-like object (has .read)
        # We use 'type: ignore' here because Pylance doesn't trust 'hasattr' enough
        if hasattr(input_data, "read"):
            if hasattr(input_data, "seek"): 
                input_data.seek(0) # type: ignore
            return input_data.read() # type: ignore
            
        # Fail safe: return empty bytes
        return b""

    def remove_background(self, image_input: Any) -> np.ndarray:
        """
        Uses AI to strip the background.
        Returns an OpenCV image (BGR) on a pure WHITE background.
        """
        try:
            # 1. Force conversion to bytes for rembg
            input_bytes = self._to_bytes(image_input)
            
            if not input_bytes:
                raise ValueError("Empty input data")

            # 2. Use AI to remove background
            output_data = remove(input_bytes) 
            
            # 3. Explicitly tell Pylance this is bytes
            clean_bytes = cast(bytes, output_data)
            
            # 4. Convert to PIL Image
            pil_img = Image.open(io.BytesIO(clean_bytes)).convert("RGBA")
            
            # 5. Create a white background
            white_bg = Image.new("RGBA", pil_img.size, "WHITE")
            white_bg.paste(pil_img, (0, 0), pil_img)
            
            # 6. Convert to OpenCV format (BGR)
            open_cv_image = np.array(white_bg.convert("RGB"))
            
            return cast(np.ndarray, open_cv_image[:, :, ::-1])

        except Exception as e:
            print(f"Background removal failed: {e}")
            # Fallback: Return original image
            try:
                nparr = np.frombuffer(self._to_bytes(image_input), np.uint8)
                decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if decoded is None: raise ValueError("Decode failed")
                return decoded
            except:
                return np.zeros((100, 100, 3), dtype=np.uint8)

    def detect_scale(self, image_input: Any, reference_diameter_mm: float = 24.26):
        """
        Detects the coin and calculates pixels_per_mm.
        """
        image = None

        # --- Robust Image Loading ---
        try:
            # Case 1: File Path (String)
            if isinstance(image_input, str):
                image = cv2.imread(image_input)
            
            # Case 2: Bytes/Buffer/Stream/Object
            else:
                file_bytes = self._to_bytes(image_input)
                if file_bytes:
                    nparr = np.frombuffer(file_bytes, dtype=np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Image load error: {e}")

        if image is None:
            return None, 0.0

        # --- Coin Detection Logic ---
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        
        circles = cv2.HoughCircles(
            blur, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=50,
            param1=50, 
            param2=30, 
            minRadius=15, 
            maxRadius=150
        )

        pixels_per_mm = 0.0
        
        if circles is not None:
            detected_circles = np.round(circles[0, :]).astype("int")
            if len(detected_circles) > 0:
                c_x, c_y, r = detected_circles[0]
                
                # Draw for debug
                cv2.circle(image, (c_x, c_y), r, (0, 255, 0), 4)
                cv2.putText(image, "Ref", (c_x - 10, c_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                coin_width_px = r * 2
                pixels_per_mm = coin_width_px / reference_diameter_mm
        
        return image, pixels_per_mm