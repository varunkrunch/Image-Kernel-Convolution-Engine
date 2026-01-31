"""
FastAPI Backend for Image Convolution Engine
Provides REST API endpoints for image processing with convolution kernels.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import numpy as np
import io
from PIL import Image

from convolution import ConvolutionEngine, create_kernel_visualization
from kernels import Kernels


# Initialize FastAPI app
app = FastAPI(
    title="Image Convolution Engine API",
    description="Apply convolution kernels to images for various effects",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize convolution engine
engine = ConvolutionEngine(padding_mode="reflect")


# Pydantic models for request/response validation
class KernelMatrix(BaseModel):
    """Model for custom kernel matrix"""
    matrix: List[List[float]] = Field(
        ...,
        description="2D kernel matrix (must be square and odd-sized)",
        example=[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
    )


class KernelValidationResponse(BaseModel):
    """Response for kernel validation"""
    valid: bool
    message: str
    kernel_size: Optional[tuple] = None


class KernelListResponse(BaseModel):
    """Response for listing available kernels"""
    kernels: Dict[str, List[List[float]]]
    count: int


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Image Convolution Engine API",
        "version": "1.0.0",
        "endpoints": {
            "POST /transform/convolution": "Apply convolution to an image",
            "GET /kernels/list": "List all available pre-built kernels",
            "POST /kernels/custom": "Validate a custom kernel",
            "GET /health": "Health check endpoint"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "convolution-engine"}


@app.get("/kernels/list", response_model=KernelListResponse)
async def list_kernels():
    """
    Get all available pre-built kernels.
    
    Returns:
        Dictionary of kernel names and their matrices
    """
    try:
        kernels = Kernels.list_kernels()
        return KernelListResponse(
            kernels=kernels,
            count=len(kernels)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing kernels: {str(e)}")


@app.post("/kernels/custom", response_model=KernelValidationResponse)
async def validate_custom_kernel(kernel: KernelMatrix):
    """
    Validate a custom kernel matrix.
    
    Args:
        kernel: Custom kernel matrix
    
    Returns:
        Validation result with details
    """
    try:
        is_valid = Kernels.validate_custom_kernel(kernel.matrix)
        
        if is_valid:
            kernel_array = np.array(kernel.matrix)
            return KernelValidationResponse(
                valid=True,
                message="Kernel is valid and ready to use",
                kernel_size=kernel_array.shape
            )
        else:
            return KernelValidationResponse(
                valid=False,
                message="Invalid kernel. Must be square, odd-sized (3x3, 5x5, etc.), and between 3x3 and 11x11",
                kernel_size=None
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error validating kernel: {str(e)}")


@app.post("/transform/convolution")
async def apply_convolution(
    file: UploadFile = File(..., description="Image file (PNG, JPG, JPEG)"),
    kernel_name: Optional[str] = Form(None, description="Pre-built kernel name"),
    custom_kernel: Optional[str] = Form(None, description="Custom kernel as JSON string")
):
    """
    Apply convolution kernel to an uploaded image.
    
    Args:
        file: Image file to process
        kernel_name: Name of pre-built kernel (e.g., 'blur', 'sharpen')
        custom_kernel: Custom kernel matrix as JSON string (e.g., '[[0,-1,0],[-1,5,-1],[0,-1,0]]')
    
    Returns:
        Processed image as PNG
    """
    # Validate file type
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PNG, JPG, and JPEG are supported"
        )
    
    # Validate file size (max 2MB as per requirements)
    contents = await file.read()
    if len(contents) > 2 * 1024 * 1024:  # 2MB in bytes
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 2MB"
        )
    
    # Determine which kernel to use
    try:
        if custom_kernel:
            # Parse custom kernel from JSON string
            import json
            kernel_matrix = json.loads(custom_kernel)
            
            # Validate custom kernel
            if not Kernels.validate_custom_kernel(kernel_matrix):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid custom kernel matrix"
                )
            
            kernel = np.array(kernel_matrix, dtype=np.float32)
            kernel_used = "custom"
        elif kernel_name:
            # Use pre-built kernel
            kernel = Kernels.get_kernel(kernel_name)
            kernel_used = kernel_name
        else:
            raise HTTPException(
                status_code=400,
                detail="Either kernel_name or custom_kernel must be provided"
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for custom kernel")
    
    # Process the image
    try:
        # Load image
        image = Image.open(io.BytesIO(contents))
        original_size = image.size
        
        # Apply convolution
        processed_bytes = engine.process_image_file(contents, kernel, optimize=True)
        
        # Return processed image
        return StreamingResponse(
            io.BytesIO(processed_bytes),
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=convolved_{file.filename}",
                "X-Kernel-Used": kernel_used,
                "X-Original-Size": f"{original_size[0]}x{original_size[1]}"
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)