"""
@author: fengyu, wangjun
@date: 20220620
@contact: fengyu_cnyc@163.com
"""

import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('sdk')

import torch

from core.model_loader.BaseModelLoader import BaseModelLoader

class FaceParsingModelLoader(BaseModelLoader):
    def __init__(self, model_path, model_category, model_name, meta_file='model_meta.json'):
        logger.info('Start to analyze the face parsing model, model path: %s, model category: %s，model name: %s' %
                    (model_path, model_category, model_name))
        super().__init__(model_path, model_category, model_name, meta_file)

        self.cfg['input_height'] = self.meta_conf['input_height']
        self.cfg['input_width'] = self.meta_conf['input_width']

        
    def load_model(self):
        device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        try:
            model = torch.load(self.cfg['model_file_path'], map_location=device, weights_only=False)
        except Exception as e:
            logger.error('The model failed to load, please check the model path: %s!'
                         % self.cfg['model_file_path'])
            raise e
        else:
            logger.info('Successfully loaded the face parsing model!')
            return model, self.cfg