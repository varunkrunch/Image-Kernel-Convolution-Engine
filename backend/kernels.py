"""
Pre-built convolution kernels for image processing.
Each kernel is a matrix that defines how pixels are transformed.
"""

import numpy as np
from typing import Dict

class Kernels:
    """Collection of standard image processing kernels"""
    
    # Emboss kernel - creates a 3D raised effect
    EMBOSS = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ], dtype=np.float32)
    
    # Box Blur kernel - averages surrounding pixels
    BLUR = np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ], dtype=np.float32)
    
    # Sharpen kernel - enhances edges and details
    SHARPEN = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    
    # Edge Detection (Sobel-like) kernel
    EDGE_DETECT = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    
    # Gaussian Blur - smoother blur than box blur
    GAUSSIAN_BLUR = np.array([
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]
    ], dtype=np.float32)
    
    # Identity kernel - returns original image (useful for testing)
    IDENTITY = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=np.float32)
    
    @classmethod
    def get_kernel(cls, name: str) -> np.ndarray:
        """Get a kernel by name"""
        kernels = {
            "emboss": cls.EMBOSS,
            "blur": cls.BLUR,
            "sharpen": cls.SHARPEN,
            "edge_detect": cls.EDGE_DETECT,
            "gaussian_blur": cls.GAUSSIAN_BLUR,
            "identity": cls.IDENTITY
        }
        
        if name.lower() not in kernels:
            raise ValueError(f"Unknown kernel: {name}. Available: {list(kernels.keys())}")
        
        return kernels[name.lower()]
    
    @classmethod
    def list_kernels(cls) -> Dict[str, list]:
        """Return all available kernels with their matrices"""
        return {
            "emboss": cls.EMBOSS.tolist(),
            "blur": cls.BLUR.tolist(),
            "sharpen": cls.SHARPEN.tolist(),
            "edge_detect": cls.EDGE_DETECT.tolist(),
            "gaussian_blur": cls.GAUSSIAN_BLUR.tolist(),
            "identity": cls.IDENTITY.tolist()
        }
    
    @classmethod
    def validate_custom_kernel(cls, kernel: list) -> bool:
        """Validate a custom kernel matrix"""
        try:
            kernel_array = np.array(kernel, dtype=np.float32)
            
            # Check if it's a 2D array
            if kernel_array.ndim != 2:
                return False
            
            # Check if it's square
            if kernel_array.shape[0] != kernel_array.shape[1]:
                return False
            
            # Check if size is odd (3x3, 5x5, 7x7, etc.)
            if kernel_array.shape[0] % 2 == 0:
                return False
            
            # Check if size is reasonable (3x3 to 11x11)
            if kernel_array.shape[0] < 3 or kernel_array.shape[0] > 11:
                return False
            
            return True
        except:
            return False