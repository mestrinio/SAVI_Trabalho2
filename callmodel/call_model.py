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
import main



def Call_Md_2d(inputs):

    labels = ['apple', 'ball', 'banana', 'bell pepper', 'binder', 'bowl', 'calculator',
               'camera', 'cap', 'cell phone', 'cereal box', 'coffee mug', 'comb', 'dry battery',
                'flashlight', 'food bag', 'food box', 'food can', 'food cup', 'food jar',
                'garlic', 'glue stick', 'greens', 'hand towel', 'instant noodles', 'keyboard',
                'kleenex', 'lemon', 'lightbulb', 'lime', 'marker', 'mushroom', 'notebook',
                'onion', 'orange', 'peach', 'pear', 'pitcher', 'plate', 'pliers', 'potato',
                'rubber eraser', 'scissors', 'shampoo', 'soda can', 'sponge', 'stapler', 
                'tomato', 'toothbrush', 'toothpaste', 'water bottle']
    
    
    test_dataset = Dataset(inputs)

    batch_size = len(inputs)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

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

    labels_gt_np = labels_gt.cpu().detach().numpy()
    ground_truths = [ [] for i in range(51)]
    for i, ground_truth in enumerate(ground_truths):
        for label in labels_gt_np:
            ground_truth.append(label == i )
    
    label_gt = []
    for idx,x in enumerate(ground_truths):
        #  print(x)
         for i in x:
              if i == True:
                   label_gt.append(labels[idx])


    
    return label_gt,label_res

with open('callmodel/files_from_scene.json', 'r') as f:
        # Reading from json file
        dataset_filenames = json.load(f)
        test_file = dataset_filenames['test_filenames']
# for  path in (paths):
label_gt,label = Call_Md_2d(test_file)
print('labels',label_gt,'probs',label)
