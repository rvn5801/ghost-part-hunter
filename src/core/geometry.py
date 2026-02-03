import numpy as np
from stl import mesh
from pathlib import Path
from typing import Tuple, Optional
import logging

# Configure local logger
logger = logging.getLogger(__name__)

class GeometryEngine:
    """
    Analyzes 3D mesh files to extract physical properties.
    """

    @staticmethod
    def get_bounding_box(file_path: Path) -> Optional[Tuple[float, float, float]]:
        """
        Calculates the axis-aligned bounding box (X, Y, Z) of an STL file in millimeters.
        
        Args:
            file_path (Path): Path to the .stl file.
            
        Returns:
            Tuple[float, float, float]: (width_x, depth_y, height_z) or None if failed.
        """
        try:
            # Load the STL file
            stl_mesh = mesh.Mesh.from_file(str(file_path))
            
            # Calculate min and max coordinates
            # points are usually shape (N, 9) (3 vertices per face), flatten to (N*3, 3)
            all_points = stl_mesh.points.reshape([-1, 3])
            
            min_coords = np.min(all_points, axis=0)
            max_coords = np.max(all_points, axis=0)
            
            # Calculate dimensions
            dims = max_coords - min_coords
            
            # Unpack dimensions (width, depth, height)
            # We assume the model is oriented somewhat correctly, 
            # but for search we often sort these later (e.g., smallest, middle, largest dim)
            width, depth, height = float(dims[0]), float(dims[1]), float(dims[2])
            
            return width, depth, height

        except Exception as e:
            logger.error(f"Failed to process geometry for {file_path.name}: {e}")
            return None

    @staticmethod
    def get_volume(file_path: Path) -> float:
        """
        Calculates the volume of the mesh (useful for filtering noise).
        """
        try:
            stl_mesh = mesh.Mesh.from_file(str(file_path))
            return float(stl_mesh.get_mass_properties()[0])
        except Exception:
            return 0.0