import os
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics import Precision, Recall, F1Score
from torchmetrics.classification import MulticlassPrecision
import json
from model import Model
from dataset_random import Dataset
from torchvision import transforms
import re
from main import main_
from PIL import Image



def Call_Md_2d(inputs = "callmodel/files_from_scene.json" ):

    labels = ['apple', 'ball', 'banana', 'bell pepper', 'binder', 'bowl', 'calculator',
               'camera', 'cap', 'cell phone', 'cereal box', 'coffee mug', 'comb', 'dry battery',
                'flashlight', 'food bag', 'food box', 'food can', 'food cup', 'food jar',
                'garlic', 'glue stick', 'greens', 'hand towel', 'instant noodles', 'keyboard',
                'kleenex', 'lemon', 'lightbulb', 'lime', 'marker', 'mushroom', 'notebook',
                'onion', 'orange', 'peach', 'pear', 'pitcher', 'plate', 'pliers', 'potato',
                'rubber eraser', 'scissors', 'shampoo', 'soda can', 'sponge', 'stapler', 
                'tomato', 'toothbrush', 'toothpaste', 'water bottle']
    
    with open(inputs, 'r') as f:
        # Reading from json file
        dataset_filenames = json.load(f)
        test_file = dataset_filenames['test_filenames']
    

    test_dataset = Dataset(test_file)

    batch_size = len(test_file)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        # Just for testing the train_loader
    tensor_to_pil_image = transforms.ToPILImage()

    # Create an instance of the model
    loaded_model = Model()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


    # Load the saved state_dict into the model
    checkpoint = torch.load("models/checkpoint_t2.pkl")
    loaded_model.load_state_dict(checkpoint["model_state_dict"])

    # Set the model to evaluation mode (important if using BatchNorm or Dropout)
    loaded_model.to(device)
    loaded_model.eval()
    for batch_idx, (inputs_, labels_gt) in enumerate(test_loader):

        # move tensors to device
        inputs_ = inputs_.to(device)
        labels_gt = labels_gt.to(device)
        print(labels_gt)

        # Get predicted labels
        labels_predicted = loaded_model.forward(inputs_)

    
    # Transform predicted labels into probabilities
    predicted_probabilities = F.softmax(labels_predicted, dim=1).tolist()
    # print(' predicted' + str(predicted_probabilities))
    probabilities = [ []  for i in range(51)]
    # probabilities_dog = [x[0] for x in predicted_probabilities]
    for x in predicted_probabilities:
        for i, probabilitie in enumerate(probabilities):
            
            probabilitie.append(x[i] > 0.95 )
    
    labels_gt_np = labels_predicted.cpu().detach().numpy()

    label_res = []
    for idx,x in enumerate(probabilities):
        #  print(x)
         for i in x:
              if i == True:
                   label_res.append(labels[idx])
    labs= []
    labs1 =[]
    lab_s = ['','','','','']
    patern='([0-9, ])+(?=])'
    patern1 = '([0-9])+'
    lab_str = str(labels_gt)
    
    for i in range(5):
        match = re.search(patern,lab_str)
        lab_ = match.group()
        match_=re.search(patern1,lab_)
        lab_s[i] = match_.group()
        lab_1 = lab_.split(',')
        labs.append(int(lab_1[i]))
    for i,y in enumerate(labs):
        labs1.append(labels[y])

    label_gt = {'0':labs1,'1':''}
    labels_gt_np = labels_gt.cpu().detach().numpy()
    ground_truths = [ [] for i in range(51)]        
    for idx,x in enumerate(ground_truths):
        #  print(x)
         for i in x:
              if i == True:
                   label_gt['1'].append(labels[idx])


    
    return label_gt,label_res


label_gt,label = Call_Md_2d()
print('labels',label_gt,'probs',label)
