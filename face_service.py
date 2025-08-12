import cv2
import torch
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class FaceIDService:
    def __init__(self):
        """Initialize the FaceID service with InsightFace model."""
        self.app = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the InsightFace model."""
        try:
            self.app = FaceAnalysis(
                name="buffalo_l", 
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("FaceID model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FaceID model: {e}")
            raise
    
    def extract_face_embeddings(self, image_bytes: bytes) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        """
        Extract face embeddings and aligned face image from uploaded image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Tuple of (face_embeddings, aligned_face_image)
            
        Raises:
            ValueError: If no face is detected or multiple faces are found
        """
        # Convert bytes to cv2 image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Invalid image format")
        
        # Detect faces
        faces = self.app.get(image)
        
        if len(faces) == 0:
            raise ValueError("No face detected in the image")
        
        if len(faces) > 1:
            logger.warning(f"Multiple faces detected ({len(faces)}), using the first one")
        
        # Extract face embeddings
        face = faces[0]
        faceid_embeds = torch.from_numpy(face.normed_embedding).unsqueeze(0)
        
        # Extract aligned face image for IP-Adapter-FaceID-Plus
        try:
            face_image = face_align.norm_crop(image, landmark=face.kps, image_size=224)
        except Exception as e:
            logger.warning(f"Failed to extract aligned face image: {e}")
            face_image = None
        
        return faceid_embeds, face_image
    
    def extract_multiple_face_embeddings(self, image_list: list[bytes]) -> torch.Tensor:
        """
        Extract face embeddings from multiple images for portrait generation.
        
        Args:
            image_list: List of image bytes
            
        Returns:
            Concatenated face embeddings tensor
        """
        faceid_embeds_list = []
        
        for image_bytes in image_list:
            embeddings, _ = self.extract_face_embeddings(image_bytes)
            faceid_embeds_list.append(embeddings.unsqueeze(0))
        
        # Concatenate along dimension 1 for portrait mode
        faceid_embeds = torch.cat(faceid_embeds_list, dim=1)
        return faceid_embeds
