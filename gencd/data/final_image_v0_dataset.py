import cv2
import glob
import importlib
import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from .final_image_dataset import FinalImageDataset
from .transforms import ComposeList

class FinalImagev0Dataset(FinalImageDataset):    
    ## override this to define self.transform
    def prepare_transforms(self):
        self.transform = None
        if self.phase == 'train':
            tfm = [
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),

                #A.CropNonEmptyMaskIfExists(self.opt.patch_size, self.opt.patch_size, p=1),
                #A.RandomCrop(self.opt.patch_size, self.opt.patch_size, p=1),
                #A.OneOf([
                #    A.CropNonEmptyMaskIfExists(self.opt.patch_size, self.opt.patch_size, p=1),
                #    A.RandomCrop(self.opt.patch_size, self.opt.patch_size, p=1),
                #], p=1.0),                   
            ]
            
            if self.opt.patch_resize_factor != 1:
                hw = int(self.opt.patch_size/self.opt.patch_resize_factor)
                tfm += [
                    A.Resize(hw, hw, interpolation=cv2.INTER_CUBIC, p=1),
                ]
            
            tfm += [ToTensorV2(p=1.0, transpose_mask=True)]
            
            self.transform = A.Compose(tfm, bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))            
        else:
            self.transform = A.Compose([
                ToTensorV2(p=1.0, transpose_mask=True)], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'])
            )    
    
