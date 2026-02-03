import cv2
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class CoinDetector:
    # US Quarter diameter in mm
    REF_DIAMETER_MM = 24.26
    
    @staticmethod
    def get_scale_factor(image_bytes: bytes) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """
        Detects a coin and returns (pixels_per_mm, processed_image).
        """
        try:
            # Convert bytes to OpenCV Image
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return None, None

            # Resize for speed (keep aspect ratio)
            height, width = img.shape[:2]
            max_dim = 800
            scale = 1.0
            if max(height, width) > max_dim:
                scale = max_dim / max(height, width)
                img = cv2.resize(img, None, fx=scale, fy=scale)
            
            # Preprocessing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Blur reduces noise (false circles)
            gray = cv2.medianBlur(gray, 5)
            
            # Hough Circle Transform
            circles = cv2.HoughCircles(
                gray, 
                cv2.HOUGH_GRADIENT, 
                dp=1.2, 
                minDist=50,  # Minimum distance between centers
                param1=50,   # Edge detection threshold
                param2=30,   # Accumulator threshold (lower = more circles)
                minRadius=20, 
                maxRadius=100
            )
            
            pixels_per_mm = None
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                # Assume the largest circle is the coin (simplistic but works for MVP)
                # Sort by radius (largest last)
                circles = circles[circles[:, 2].argsort()]
                largest_circle = circles[-1]
                
                (x, y, r) = largest_circle
                
                # Draw the detected coin for debugging
                cv2.circle(img, (x, y), r, (0, 255, 0), 4)
                cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
                
                # Calculate Scale
                diameter_pixels = r * 2
                pixels_per_mm = (diameter_pixels / CoinDetector.REF_DIAMETER_MM) / scale
                
                logger.info(f"Coin detected! Diameter: {diameter_pixels}px -> {pixels_per_mm:.2f} px/mm")

            return pixels_per_mm, img

        except Exception as e:
            logger.error(f"Vision Error: {e}")
            return None, None