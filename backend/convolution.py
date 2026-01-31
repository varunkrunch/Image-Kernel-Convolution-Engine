"""
Core convolution engine - implements image convolution from scratch using NumPy.
NO OpenCV shortcuts - pure mathematical implementation.
"""

import numpy as np
from PIL import Image
from typing import Tuple, Union
import io


class ConvolutionEngine:
    """
    Implements 2D convolution for image processing from scratch.
    
    The convolution operation slides a kernel (small matrix) over an image,
    multiplying overlapping values and summing them to produce a new pixel value.
    """
    
    def __init__(self, padding_mode: str = "reflect"):
        """
        Initialize the convolution engine.
        
        Args:
            padding_mode: How to handle image borders
                - 'reflect': Mirror pixels at the border
                - 'edge': Extend edge pixels
                - 'constant': Use zeros for padding
        """
        self.padding_mode = padding_mode
    
    def apply_convolution(
        self, 
        image: Union[Image.Image, np.ndarray], 
        kernel: np.ndarray
    ) -> Image.Image:
        """
        Apply convolution kernel to an image.
        
        Args:
            image: PIL Image or numpy array
            kernel: 2D numpy array (convolution kernel)
        
        Returns:
            PIL Image with convolution applied
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Store original dtype and handle different image formats
        original_dtype = img_array.dtype
        
        # Convert to float for processing
        img_array = img_array.astype(np.float32)
        
        # Handle grayscale vs RGB
        if len(img_array.shape) == 2:
            # Grayscale image
            result = self._convolve_2d(img_array, kernel)
        elif len(img_array.shape) == 3:
            # RGB or RGBA image - apply convolution to each channel
            channels = []
            for i in range(img_array.shape[2]):
                channel = img_array[:, :, i]
                convolved = self._convolve_2d(channel, kernel)
                channels.append(convolved)
            result = np.stack(channels, axis=2)
        else:
            raise ValueError(f"Unsupported image shape: {img_array.shape}")
        
        # Clip values to valid range and convert back
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        return Image.fromarray(result)
    
    def _convolve_2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Perform 2D convolution on a single channel.
        
        This is the core algorithm - implemented from scratch using only NumPy.
        
        Args:
            image: 2D numpy array (single channel)
            kernel: 2D numpy array (convolution kernel)
        
        Returns:
            Convolved 2D numpy array
        """
        # Get dimensions
        img_height, img_width = image.shape
        kernel_height, kernel_width = kernel.shape
        
        # Kernel must be odd-sized and square for this implementation
        assert kernel_height % 2 == 1 and kernel_width % 2 == 1, \
            "Kernel dimensions must be odd"
        assert kernel_height == kernel_width, \
            "Kernel must be square"
        
        # Calculate padding needed
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2
        
        # Pad the image to handle borders
        padded_image = self._pad_image(image, pad_height, pad_width)
        
        # Initialize output array
        output = np.zeros_like(image, dtype=np.float32)
        
        # Perform convolution using sliding window
        # This is the actual convolution operation
        for i in range(img_height):
            for j in range(img_width):
                # Extract the region of interest (same size as kernel)
                region = padded_image[i:i+kernel_height, j:j+kernel_width]
                
                # Element-wise multiplication and sum (this is convolution!)
                # Flip kernel for true convolution (vs correlation)
                output[i, j] = np.sum(region * np.flip(kernel))
        
        return output
    
    def _pad_image(
        self, 
        image: np.ndarray, 
        pad_height: int, 
        pad_width: int
    ) -> np.ndarray:
        """
        Pad image to handle border pixels during convolution.
        
        Args:
            image: 2D numpy array
            pad_height: Padding rows top and bottom
            pad_width: Padding columns left and right
        
        Returns:
            Padded image
        """
        if self.padding_mode == "reflect":
            # Mirror the image at borders
            return np.pad(
                image, 
                ((pad_height, pad_height), (pad_width, pad_width)),
                mode='reflect'
            )
        elif self.padding_mode == "edge":
            # Extend edge values
            return np.pad(
                image,
                ((pad_height, pad_height), (pad_width, pad_width)),
                mode='edge'
            )
        elif self.padding_mode == "constant":
            # Use zeros for padding
            return np.pad(
                image,
                ((pad_height, pad_height), (pad_width, pad_width)),
                mode='constant',
                constant_values=0
            )
        else:
            raise ValueError(f"Unknown padding mode: {self.padding_mode}")
    
    def optimized_convolve_2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Optimized version using NumPy's vectorization capabilities.
        Still implements convolution from scratch, but faster.
        
        Args:
            image: 2D numpy array (single channel)
            kernel: 2D numpy array (convolution kernel)
        
        Returns:
            Convolved 2D numpy array
        """
        img_height, img_width = image.shape
        kernel_height, kernel_width = kernel.shape
        
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2
        
        padded_image = self._pad_image(image, pad_height, pad_width)
        output = np.zeros_like(image, dtype=np.float32)
        
        # Flip kernel once for true convolution
        flipped_kernel = np.flip(kernel)
        
        # Vectorized approach - still doing convolution from scratch
        for i in range(kernel_height):
            for j in range(kernel_width):
                # Shift and multiply
                shifted = padded_image[i:i+img_height, j:j+img_width]
                output += shifted * flipped_kernel[i, j]
        
        return output
    
    def process_image_file(
        self, 
        image_bytes: bytes, 
        kernel: np.ndarray,
        optimize: bool = True
    ) -> bytes:
        """
        Process image from bytes and return processed image as bytes.
        
        Args:
            image_bytes: Input image as bytes
            kernel: Convolution kernel
            optimize: Use optimized convolution method
        
        Returns:
            Processed image as bytes
        """
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Apply convolution
        if optimize:
            # Use optimized version for better performance
            img_array = np.array(image).astype(np.float32)
            
            if len(img_array.shape) == 2:
                result = self.optimized_convolve_2d(img_array, kernel)
            elif len(img_array.shape) == 3:
                channels = []
                for i in range(img_array.shape[2]):
                    channel = img_array[:, :, i]
                    convolved = self.optimized_convolve_2d(channel, kernel)
                    channels.append(convolved)
                result = np.stack(channels, axis=2)
            
            result = np.clip(result, 0, 255).astype(np.uint8)
            processed_image = Image.fromarray(result)
        else:
            processed_image = self.apply_convolution(image, kernel)
        
        # Convert to bytes
        output_buffer = io.BytesIO()
        processed_image.save(output_buffer, format='PNG')
        output_buffer.seek(0)
        
        return output_buffer.getvalue()


def create_kernel_visualization(kernel: np.ndarray) -> str:
    """
    Create a text visualization of the kernel matrix.
    
    Args:
        kernel: 2D numpy array
    
    Returns:
        Formatted string representation
    """
    lines = []
    for row in kernel:
        formatted_row = [f"{val:7.3f}" for val in row]
        lines.append("[" + ", ".join(formatted_row) + "]")
    
    return "\n".join(lines)