"""
@author: Jun Wang
@date: 20201101
@contact: jun21wangustc@gmail.com
"""

import os
import logging as logger
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger.basicConfig(level=logger.INFO, 
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

class CommonTestDataset(Dataset):
    """ Data processor for model evaluation.

    Attributes:
        image_root(str): root directory of test set.
        image_list_file(str): path of the image list file.
        crop_eye(bool): crop eye(upper face) as input or not.
    """
    def __init__(self, image_root, image_list_file, crop_eye=False):
        self.image_root = image_root
        self.image_list = []
        
        with open(image_list_file) as f:
            for line in f:
                line = line.strip().replace('\\', '/')
                if not line:
                    continue
                
                filename = os.path.basename(line)  # np. 10032527@N_0000.jpg
                folder_name = os.path.basename(self.image_root)  # np. 10032527@N
                print("line[0:10]:", line[0:10], "line[11:21]" + line[11:21])  # Debugging output
                if line[0:10] == line[11:21]:
                    self.image_list.append(line)
        self.mean = 127.5
        self.std = 128.0
        self.crop_eye = crop_eye
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, index):
        short_image_path = self.image_list[index]
        image_path = os.path.join(self.image_root, short_image_path)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        #image = cv2.resize(image, (128, 128))
        if self.crop_eye:
            image = image[:60, :]
        image = (image.transpose((2, 0, 1)) - self.mean) / self.std
        image = torch.from_numpy(image.astype(np.float32))
        return image, short_image_path
