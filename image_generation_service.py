import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
import numpy as np
from typing import List, Optional
import logging
import os

logger = logging.getLogger(__name__)

class ImageGenerationService:
    def __init__(self, device: str = "cuda"):
        """Initialize the image generation service."""
        self.device = device if torch.cuda.is_available() else "cpu"
        self.base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
        self.vae_model_path = "stabilityai/sd-vae-ft-mse"
        self.ip_adapter = None
        self.pipe = None
        
        # Download URLs for IP-Adapter models
        self.model_urls = {
            "faceid": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin",
            "faceid_plus": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plus_sd15.bin",
            "faceid_plusv2": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin",
            "faceid_portrait": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait_sd15.bin"
        }
        
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Setup the diffusion pipeline."""
        try:
            logger.info("Setting up diffusion pipeline...")
            
            # Setup scheduler
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
            )
            
            # Setup VAE
            vae = AutoencoderKL.from_pretrained(self.vae_model_path).to(dtype=torch.float16)
            
            # Setup pipeline
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                scheduler=noise_scheduler,
                vae=vae,
                feature_extractor=None,
                safety_checker=None
            )
            
            self.pipe = self.pipe.to(self.device)
            
            logger.info("Diffusion pipeline setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup pipeline: {e}")
            raise
    
    def _download_ip_adapter_model(self, model_type: str = "faceid") -> str:
        """Download IP-Adapter model if not exists."""
        import requests
        
        model_filename = f"ip-adapter-{model_type}_sd15.bin"
        model_path = os.path.join("models", model_filename)
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        if not os.path.exists(model_path):
            logger.info(f"Downloading IP-Adapter model: {model_type}")
            url = self.model_urls.get(model_type)
            if not url:
                raise ValueError(f"Unknown model type: {model_type}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Model downloaded: {model_path}")
        
        return model_path
    
    def load_ip_adapter(self, model_type: str = "faceid"):
        """Load IP-Adapter model."""
        try:
            # Import IP-Adapter classes (you'll need to install ip-adapter package)
            from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
            
            model_path = self._download_ip_adapter_model(model_type)
            
            if model_type == "faceid_portrait":
                self.ip_adapter = IPAdapterFaceID(
                    self.pipe, model_path, self.device, 
                    num_tokens=16, n_cond=5
                )
            else:
                self.ip_adapter = IPAdapterFaceID(self.pipe, model_path, self.device)
            
            logger.info(f"IP-Adapter {model_type} loaded successfully")
            
        except ImportError:
            logger.error("IP-Adapter package not found. Please install it from https://github.com/tencent-ailab/IP-Adapter")
            raise
        except Exception as e:
            logger.error(f"Failed to load IP-Adapter: {e}")
            raise
    
    def generate_images(
        self,
        prompt: str,
        faceid_embeds: torch.Tensor,
        negative_prompt: str = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry",
        num_samples: int = 1,
        width: int = 512,
        height: int = 768,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Generate images using IP-Adapter-FaceID.
        
        Args:
            prompt: Text prompt for image generation
            faceid_embeds: Face embeddings tensor
            negative_prompt: Negative prompt
            num_samples: Number of images to generate
            width: Image width
            height: Image height
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            seed: Random seed for reproducibility
            
        Returns:
            List of generated PIL Images
        """
        if self.ip_adapter is None:
            raise ValueError("IP-Adapter not loaded. Call load_ip_adapter() first.")
        
        try:
            # Set seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            # Generate images
            images = self.ip_adapter.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                faceid_embeds=faceid_embeds,
                num_samples=num_samples,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed
            )
            
            logger.info(f"Generated {len(images)} images successfully")
            return images
            
        except Exception as e:
            logger.error(f"Failed to generate images: {e}")
            raise
    
    def generate_images_plus(
        self,
        prompt: str,
        faceid_embeds: torch.Tensor,
        face_image: np.ndarray,
        negative_prompt: str = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry",
        num_samples: int = 1,
        width: int = 512,
        height: int = 768,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        s_scale: float = 1.0,
        shortcut: bool = False,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Generate images using IP-Adapter-FaceID-Plus.
        
        Args:
            prompt: Text prompt for image generation
            faceid_embeds: Face embeddings tensor
            face_image: Aligned face image
            negative_prompt: Negative prompt
            num_samples: Number of images to generate
            width: Image width
            height: Image height
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            s_scale: Scale for face structure
            shortcut: Whether to use v2 shortcut
            seed: Random seed for reproducibility
            
        Returns:
            List of generated PIL Images
        """
        if self.ip_adapter is None:
            raise ValueError("IP-Adapter not loaded. Call load_ip_adapter() first.")
        
        try:
            # Set seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            # Generate images with Plus model
            images = self.ip_adapter.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                face_image=face_image,
                faceid_embeds=faceid_embeds,
                shortcut=shortcut,
                s_scale=s_scale,
                num_samples=num_samples,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed
            )
            
            logger.info(f"Generated {len(images)} images using Plus model")
            return images
            
        except Exception as e:
            logger.error(f"Failed to generate images with Plus model: {e}")
            raise
