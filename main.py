from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import io
import base64
import logging
from contextlib import asynccontextmanager

from face_service import FaceIDService
from image_generation_service import ImageGenerationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global services
face_service: Optional[FaceIDService] = None
image_gen_service: Optional[ImageGenerationService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - initialize services on startup."""
    global face_service, image_gen_service
    
    try:
        logger.info("Initializing services...")
        
        # Initialize services
        face_service = FaceIDService()
        image_gen_service = ImageGenerationService()
        
        # Load default IP-Adapter model
        image_gen_service.load_ip_adapter("faceid")
        
        logger.info("Services initialized successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    finally:
        logger.info("Shutting down services...")

# Create FastAPI app
app = FastAPI(
    title="IP-Adapter FaceID Image Generator",
    description="Generate consistent pictures of people using IP-Adapter-FaceID and Realistic Vision V4.0",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: str = Field(
        default="monochrome, lowres, bad anatomy, worst quality, low quality, blurry",
        description="Negative prompt"
    )
    num_samples: int = Field(default=1, ge=1, le=4, description="Number of images to generate")
    width: int = Field(default=512, ge=256, le=1024, description="Image width")
    height: int = Field(default=768, ge=256, le=1024, description="Image height")
    num_inference_steps: int = Field(default=30, ge=10, le=100, description="Number of denoising steps")
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0, description="Guidance scale")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")

class GenerationPlusRequest(GenerationRequest):
    s_scale: float = Field(default=1.0, ge=0.0, le=2.0, description="Scale for face structure")
    shortcut: bool = Field(default=False, description="Use v2 shortcut")

class GenerationResponse(BaseModel):
    success: bool
    images: List[str]  # Base64 encoded images
    message: str

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "IP-Adapter FaceID Image Generator API",
        "version": "1.0.0",
        "endpoints": {
            "/generate": "Generate images using IP-Adapter-FaceID (auto-detects Plus mode)",
            "/health": "Health check endpoint"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "face_service": face_service is not None,
        "image_gen_service": image_gen_service is not None
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_images(
    face_image: UploadFile = File(..., description="Face image for ID extraction"),
    prompt: str = Form(..., description="Text prompt for image generation"),
    negative_prompt: str = Form(
        default="monochrome, lowres, bad anatomy, worst quality, low quality, blurry",
        description="Negative prompt"
    ),
    num_samples: int = Form(default=1, ge=1, le=4, description="Number of images to generate"),
    width: int = Form(default=512, ge=256, le=1024, description="Image width"),
    height: int = Form(default=768, ge=256, le=1024, description="Image height"),
    num_inference_steps: int = Form(default=30, ge=10, le=100, description="Number of denoising steps"),
    guidance_scale: float = Form(default=7.5, ge=1.0, le=20.0, description="Guidance scale"),
    s_scale: Optional[float] = Form(default=None, ge=0.0, le=2.0, description="Face structure scale (enables Plus mode if provided)"),
    shortcut: bool = Form(default=False, description="Use PlusV2 model (only applies in Plus mode)"),
    seed: Optional[int] = Form(default=None, description="Random seed for reproducibility")
):
    """Generate images using IP-Adapter-FaceID or FaceID-Plus (auto-detected based on parameters)."""
    try:
        # Validate services
        if not face_service or not image_gen_service:
            raise HTTPException(status_code=503, detail="Services not initialized")
        
        # Validate file type
        if not face_image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await face_image.read()
        
        # Extract face embeddings and aligned face image
        faceid_embeds, face_img = face_service.extract_face_embeddings(image_data)
        
        # Determine if we should use Plus mode
        use_plus_mode = s_scale is not None
        
        if use_plus_mode:
            # Plus mode: requires aligned face image
            if face_img is None:
                raise HTTPException(status_code=400, detail="Failed to extract aligned face image for Plus mode")
            
            # Load appropriate Plus model
            model_type = "faceid_plusv2" if shortcut else "faceid_plus"
            image_gen_service.load_ip_adapter(model_type)
            
            # Generate with Plus model
            generated_images = image_gen_service.generate_images_plus(
                prompt=prompt,
                faceid_embeds=faceid_embeds,
                face_image=face_img,
                negative_prompt=negative_prompt,
                num_samples=num_samples,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                s_scale=s_scale,
                shortcut=shortcut,
                seed=seed
            )
            model_name = "Plus"
        else:
            # Standard mode
            image_gen_service.load_ip_adapter("faceid")
            
            # Generate with standard model
            generated_images = image_gen_service.generate_images(
                prompt=prompt,
                faceid_embeds=faceid_embeds,
                negative_prompt=negative_prompt,
                num_samples=num_samples,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed
            )
            model_name = "Standard"
        
        # Convert images to base64
        image_b64_list = []
        for img in generated_images:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            image_b64_list.append(img_b64)
        
        return GenerationResponse(
            success=True,
            images=image_b64_list,
            message=f"Successfully generated {len(generated_images)} images using {model_name} model"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in generate_images: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")



if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
