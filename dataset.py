# -*- coding: utf-8 -*-
import os
import torch.utils.data as data
import random

from PIL import Image
from torchvision import transforms
# Define the format for images
def default_loader(path):
    return Image.open(path).convert('RGB')

class RafDataset(data.Dataset):
    def __init__(self, path, phase,label_path, transform=None,loader = default_loader):
        self.raf_path = path
        self.phase = phase
        self.label_path = label_path
        self.transform = transform
        self.loader = loader
        fh = open(os.path.join( label_path), 'r', encoding='utf-8')
        self.label = []
        self.images_names=[]
        # Open this text with the path and txt text parameter passed in, using read-only mode
        for line in fh: 
            line = line.strip('\n')
            line = line.rstrip('\n') 
            words = line.split() 
            if(words[0].startswith(phase)):
                self.images_names.append(words[0])
                self.label.append(int(words[1]))
        # self.file_paths = []

        # for f in images_names:
        #     f = f.split(".")[0]
        #     f += '_aligned.jpg'
        #     file_name = os.path.join(self.raf_path, 'Image/aligned', f)
        #     self.file_paths.append(file_name)

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        label = self.label[idx]-1
        img_path1 = self.images_names[idx]
        img_path = os.path.join(self.raf_path,img_path1)
        image = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(image)  # Convert Data Labels to Tensor

        return img, label,img_path1