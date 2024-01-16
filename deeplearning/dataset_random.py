#!/usr/bin/env python3


import os
import torch
from torchvision import transforms
from PIL import Image
import re
from labels import switch


pattern = '([a-z_]+)(?=_\d)'

class Dataset(torch.utils.data.Dataset):

    def __init__(self, filenames):
        self.filenames = filenames
        self.number_of_images = len(self.filenames)

        # Compute the corresponding labels
        # self.labels should be like ['cat', 'dog', 'cat'], but we will use [1, 0, 1] because of pytorch
        # self.labels = []
        # for filename in self.filenames:
        #     basename = os.path.basename(filename)
        #     blocks = basename.split('.')
        #     label = blocks[0]  # because basename is "cat.2109.jpg"

        #     if label == 'dog':
        #         self.labels.append(0)
        #     elif label == 'cat':
        #         self.labels.append(1)
        #     else:
        #         raise ValueError('Unknown label ' + label)
        self.name = []
        
        for filename in self.filenames:
            match = re.search(pattern, filename)
            label = match.group(1)
            if all(item != label for item in self.name):
                self.name.append(label)
        print(self.name)
            # else:
            #     self.name[label]=[filename]
            
                

        # files=files_+files_1
        # print(files)
        # for label_ in self.name:
        #     print(label_)
        #     apple_len= 0
        # self.labels = self.name[label]
        # self.labels = switch(self.name,len(self.name[label[1]]))
        # print (self.labels)
        # self.labels = switch(self.filenames)



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
        # print(index)
        filename = self.filenames[index]
        pil_image = Image.open(filename)

        # Convert to tensor
        tensor_image = self.transforms(pil_image)

        # Get corresponding label
        match = re.search(pattern, filename)
        label = match.group(1)
        label_num = -1
        for  idx,item in enumerate(self.name):
            if item == label:
                label_num  = idx 
                break
        # print('label =' + label + ' idx = ' ,label_num)

        # label = self.labels[index]

        return tensor_image, label_num
