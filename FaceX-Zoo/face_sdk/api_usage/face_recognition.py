"""
@author: FaceX-Zoo Contributors
"""
import sys
import yaml
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')

import numpy as np
from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler

class FaceRecognition:
    """Face recognition wrapper class"""
    
    def __init__(self, device='cpu'):
        """Initialize face recognition model
        
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
        model_category = 'face_recognition'
        model_name = model_conf[scene][model_category]
        
        # Load recognition model
        try:
            faceRecModelLoader = FaceRecModelLoader(model_path, model_category, model_name)
            model, cfg = faceRecModelLoader.load_model()
            
            # Handle model.module if needed
            if hasattr(model, 'module'):
                model = model.module
                
            if self.device == 'cpu':
                model = model.cpu()
                
            self.handler = FaceRecModelHandler(model, self.device, cfg)
        except Exception as e:
            logger.error('Failed to load face recognition model!')
            logger.error(e)
            raise e
    
    def get_feature(self, image):
        """Extract face feature/embedding from an aligned face image
        
        Args:
            image: Aligned face image (numpy.ndarray)
            
        Returns:
            numpy.ndarray: Face feature vector
        """
        try:
            feature = self.handler.inference_on_image(image)
            return feature
        except Exception as e:
            logger.error('Face feature extraction failed!')
            logger.error(e)
            raise e
            
    def compare(self, feature1, feature2):
        """Compare two face features
        
        Args:
            feature1: First face feature
            feature2: Second face feature
            
        Returns:
            float: Similarity score
        """
        return np.dot(feature1, feature2)
