"""
@author: FaceX-Zoo Contributors
"""
import sys
import yaml
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')

import numpy as np
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper

class FaceAlignment:
    """Face alignment and landmark detection wrapper class"""
    
    def __init__(self, device='cpu'):
        """Initialize face alignment model
        
        Args:
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device
        
        # Load model configuration
        with open('config/model_conf.yaml') as f:
            model_conf = yaml.load(f, Loader=yaml.FullLoader)
            
        # Common settings
        model_path = 'models'
        scene = 'non-mask'
        model_category = 'face_alignment'
        model_name = model_conf[scene][model_category]
        
        # Load alignment model
        try:
            faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
            model, cfg = faceAlignModelLoader.load_model()
            self.handler = FaceAlignModelHandler(model, self.device, cfg)
            self.cropper = FaceRecImageCropper()
        except Exception as e:
            logger.error('Failed to load face alignment model!')
            logger.error(e)
            raise e
    
    def get_landmarks(self, image, detection):
        """Get facial landmarks for a detected face
        
        Args:
            image: Input image (numpy.ndarray)
            detection: Face detection result [x1, y1, x2, y2, confidence]
            
        Returns:
            numpy.ndarray: Array of facial landmarks
        """
        try:
            # Convert detection to integer values if needed
            det = detection.astype(np.int32) if isinstance(detection, np.ndarray) else np.array(detection[:4], dtype=np.int32)
            landmarks = self.handler.inference_on_image(image, det)
            return landmarks
        except Exception as e:
            logger.error('Face landmark detection failed!')
            logger.error(e)
            raise e
    
    def align(self, image, landmarks):
        """Align and crop face using landmarks
        
        Args:
            image: Input image (numpy.ndarray)
            landmarks: Facial landmarks
            
        Returns:
            numpy.ndarray: Aligned face image
        """
        try:
            landmarks_list = []
            for (x, y) in landmarks.astype(np.int32):
                landmarks_list.extend((x, y))
            
            aligned_face = self.cropper.crop_image_by_mat(image, landmarks_list)
            return aligned_face
        except Exception as e:
            logger.error('Face alignment failed!')
            logger.error(e)
            raise e
