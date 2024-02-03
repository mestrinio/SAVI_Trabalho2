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
from deeplearning.model import Model
from deeplearning.dataset_random import Dataset
from torchvision import transforms




def Call_Md_2d(inputs):
    test_dataset = Dataset(inputs)

    batch_size = len(inputs)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

        # Just for testing the train_loader
    tensor_to_pil_image = transforms.ToPILImage()

    # Create an instance of the model
    loaded_model = Model()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


    # Load the saved state_dict into the model
    checkpoint = torch.load("models/checkpoint_t3.pkl")
    loaded_model.load_state_dict(checkpoint["model_state_dict"])

    # Set the model to evaluation mode (important if using BatchNorm or Dropout)
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

    
    return labels_predicted,probabilities

paths = ['objects_pcd/objectspng/output_image_0.png','objects_pcd/objectspng/output_image_1.png','objects_pcd/objectspng/output_image_2.png',
         'objects_pcd/objectspng/output_image_3.png','objects_pcd/objectspng/output_image_4.png']
# for  path in (paths):
labels,probs = Call_Md_2d(paths)
print(labels,probs)