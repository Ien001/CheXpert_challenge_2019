# encoding: utf-8

"""
Read images and corresponding labels.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class ChestXrayDataSet(Dataset):
    def __init__(self, image_list_file, transform=None, gray_scale = False):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        with open(image_list_file, "r") as f:
            for line in f:
                if '.jpg' in line:
                    image_names.append(line.replace('\n',''))

        self.image_names = image_names
        self.transform = transform
        self.gray_scale = gray_scale

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        if self.gray_scale:
            image = Image.open(image_name).convert('L')
        else:
            image = Image.open(image_name).convert('RGB')
            
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_names)

