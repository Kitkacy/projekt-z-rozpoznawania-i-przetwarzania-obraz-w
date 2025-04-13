"""
@author: FaceX-Zoo Contributors
"""
import sys
import yaml
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')

import numpy as np
import torch
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler

class FaceDetection:
    """Face detection wrapper class"""
    
    def __init__(self, device='cpu'):
        """Initialize face detection model
        
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
        model_category = 'face_detection'
        model_name = model_conf[scene][model_category]
        
        # Load detection model
        try:
            faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
            model, cfg = faceDetModelLoader.load_model()
            self.handler = FaceDetModelHandler(model, self.device, cfg)
        except Exception as e:
            logger.error('Failed to load face detection model!')
            logger.error(e)
            raise e
    
    def detect(self, image):
        """Detect faces in an image
        
        Args:
            image: Input image (numpy.ndarray)
            
        Returns:
            numpy.ndarray: Array of face detections [x1, y1, x2, y2, confidence]
        """
        try:
            dets = self.handler.inference_on_image(image)
            return dets
        except Exception as e:
            logger.error('Face detection failed!')
            logger.error(e)
            raise e
