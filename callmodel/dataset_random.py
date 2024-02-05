#!/usr/bin/env python3


import os
import torch
from torchvision import transforms, datasets
from PIL import Image
import re



pattern = '([a-z_]+)(?=_\d)'
pattern1 =  '([0-9]+)(?=.p)'

class Dataset(torch.utils.data.Dataset):

    def __init__(self, filenames):
        # self.transforms = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor()
        # ])
        
        # self.filenames = datasets.ImageFolder(filenames,transform=self.transforms)
        self.filenames = filenames
        self.number_of_images = len(self.filenames)
        self.b = 'files invalid'
        # Compute the corresponding labels
        # self.labels should be like ['cat', 'dog', 'cat'], but we will use [1, 0, 1] because of pytorch
        if self.number_of_images < 10:
            self.labels = []
            self.b=1
            for filename in self.filenames:
                # basename = os.path.basename(filename)
                # blocks = basename.split('.')
                # label = blocks[0]  # because basename is "cat.2109.jpg"
                match = re.search(pattern, filename)
                label = match.group(1)


                if label == 'bowl':
                    self.labels.append(5)
                elif label == 'cap':
                    self.labels.append(8)
                elif label == 'cereal':
                    self.labels.append(10)
                elif label == 'coffee':
                    self.labels.append(11)
                elif label == 'soda':
                    self.labels.append(44)
                else:
                    match = re.search(pattern1, filename)
                    label = match.group(1)
                    a = '2','4'
                    if label == '2' or label == '4' :
                        self.labels.append(5)
                    elif label == '1':
                        self.labels.append(8)
                    elif label == '':
                        self.labels.append(10)
                    elif label == '0':
                        self.labels.append(11)
                    elif label == '3':
                        self.labels.append(44)
                    else:   
                        raise ValueError('Unknown label ' + label)
        else:    
            self.name = []
            self.b=0
            for filename in self.filenames:
                match = re.search(pattern, filename)
                label = match.group(1)
                if all(item != label for item in self.name):
                    self.name.append(label)
            self.name.sort()
            print(self.name)
                # else:
                #     self.name[label]=[filename]
            
                

        # files=files_+files_1
        # print(files)
            for i,label_ in enumerate(self.name):
                print(label_)
                apple_len= 0
                self.labels.append(i)
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
        if self.b== 0:
            match = re.search(pattern, filename)
            label = match.group(1)
            label_num = -1
            for  idx,item in enumerate(self.name):
                if item == label:
                    label_num  = idx 
                    break
            print('label =' + label + ' idx = ' ,label_num)

        elif self.b == 1:
            label_num = self.labels[index]
        else:
            raise ValueError('Unknown label ' + self.b)

        return tensor_image, label_num
