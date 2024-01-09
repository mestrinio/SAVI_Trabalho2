#!/usr/bin/env python3


import os
import torch
import re
from torchvision import transforms
from PIL import Image

pattern = '([a-z_]+)(?=_\d)'

class Dataset(torch.utils.data.Dataset):

    def __init__(self, filenames):
        # self.chosen_obj = chosen_obj
        self.num=[]
        self.filenames = filenames
        self.number_of_images = len(self.filenames)

        # Compute the corresponding labels
        # self.labels should be like ['cat', 'dog', 'cat'], but we will use [1, 0, 1] because of pytorch
        self.labels = {}
        for filename in self.filenames:        
            match = re.search(pattern, filename)
            label = match.group(1)
            if any(key == label for key in self.labels.keys()):
                self.labels[label].append(filename)
            else:
                self.labels[label]=[filename]
            # basename = os.path.basename(filename)
            # blocks = basename.split('.')
            # label = blocks[0]  # because basename is "cat.2109.jpg"

            if label == 'apples':
                self.num.append(0)
            elif label != 'apples':
                self.num.append(1)
            else:
                raise ValueError('Unknown label ' + label)

        # print(self.filenames[0:3])
        # print(self.labels[0:3])
        # indexes = [0, 1, 2, ...]
        # filenames ['/home/mike/savi_datasets/dogs-vs-cats/train/cat.2832.jpg', '/home/mike/savi_datasets/dogs-vs-cats/train/cat.8274.jpg', '/home/mike/savi_datasets/dogs-vs-cats/train/cat.4537.jpg']
        # labels ['cat', 'cat', 'cat']

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        # must return the size of the data
        return self.number_of_images

    def __getitem__(self, index):
        # Must return the data of the corresponding index

        # Load the image in pil format
        filename = self.filenames[index]
        pil_image = Image.open(filename)

        # Convert to tensor
        tensor_image = self.transforms(pil_image)

        # Get corresponding label
        label = self.num[index]

        return tensor_image, label
