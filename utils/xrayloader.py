import os
import sys
import pickle
import cv2
# from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
# from skimage.transform import rotate
import torchvision.transforms as T
import json

class XrayDataset_val(Dataset):
    def __init__(self,
                 args,
                 transform=None):
        df = pd.read_csv(args.val_csv_dir)
        self.img_list = np.array(df['file_name'])
        self.label_list = np.array(df['label'])
        self.data_path = args.data_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.img_list[index]
        name = name.replace(".tiff", ".jpg")
        img_path = os.path.join(self.data_path, name)

        mask_name = self.img_list[index]
        mask_name = mask_name.replace(".tiff", ".jpg")
        msk_path = os.path.join(self.data_path.replace('images', 'mask'), mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')
        
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)
    
        label_cls = torch.tensor(self.label_list[index])
        return img, mask>0, label_cls, name