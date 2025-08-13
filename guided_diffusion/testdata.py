import torch.utils.data as data
import torchvision.transforms
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize, transforms
import re
from PIL import ImageFile
from os import path
import numpy as np
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
# --- Training dataset --- #
import torch as th
import cv2
from  torchvision import utils as vutils
class TestData(data.Dataset):
    def __init__(self, data_dir, crop_size=[256,256],isAll = False):
        super().__init__()
        self.data_dir = data_dir
      
        input_names=os.listdir(self.data_dir)
        # self.input_names=input_names[:20]
        self.input_names=input_names
        if isAll == True:
            self.input_names = input_names
        self.crop_size = crop_size
        # print(self.input_names)
        self.resolution=384

    def get_images(self, index):
        input_name = self.input_names[index]
 
        thermal_image = self.process_and_load_images(os.path.join(self.data_dir  ,input_name))

        out_dict={'thermal': thermal_image, 'Index': input_name}

        return  out_dict

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)

    def process_and_load_images_our(self,path):
        pil_image = Image.open(path).convert('RGB')
        ToTensor = torchvision.transforms.ToTensor()
        arr = ToTensor(pil_image)
        Resize = torchvision.transforms.Resize([384, 384])
        arr = Resize(arr)
        return arr

    def process_and_load_images(self,path):
        pil_image = Image.open(path).convert('RGB')
        # tf = torchvision.transforms.ToTensor()
        # pil_image = tf(pil_image)
        #
        # return pil_image.numpy()

        pil_image=pil_image.resize((self.resolution,self.resolution))
        arr=np.array(pil_image).astype(np.float32)
        arr=arr/127.5-1.0
        arr = np.transpose(arr, [2, 0, 1])
        return arr

    def process_and_load_images_BAK(self,path):
        pil_image = Image.open(path)
        pil_image=pil_image.resize((self.resolution,self.resolution))
        arr=np.array(pil_image).astype(np.float32)
        arr=arr/127.5-1.0
        if arr.ndim < 3:
            # Unified dimension
            arr = np.array([arr, arr, arr])
            return arr
        arr = np.transpose(arr, [2, 0, 1])

        return arr