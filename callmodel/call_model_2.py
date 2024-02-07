import os
import torch
import torch.nn.functional as F
#from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt
import itertools
import numpy as np
from torch.utils.data import DataLoader
#from torchmetrics import Precision, Recall, F1Score
#from torchmetrics.classification import MulticlassPrecision
import json
try:
    from model import Model
    from dataset_random import Dataset
    from main import main_

except:
    from callmodel.model import Model
    from callmodel.dataset_random import Dataset
    from callmodel.main import main_
from torchvision import transforms
import re

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
    checkpoint = torch.load("models/checkpoint_t3.pkl")
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
            
            probabilitie.append(x[i] > 0.45 )
    
   
    
    label_res = []
    for i in range(batch_size):    
        for idx,x in enumerate(probabilities):
        #  print(x)
        
            if x[i] == True:
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

   


    

    labels_gt_np = labels_gt.cpu().detach().numpy()
    ground_truths = [ [] for i in range(51)]
    for i, ground_truth in enumerate(ground_truths):
        
        for label in labels_gt_np:
            ground_truth.append(label == i )

    # ground_truth_is_dog = [x == 0 for x in labels_gt_np]
    # print('ground_truth_is_dog=' + str(ground_truth_is_dog))

    # labels_predicted_np = labels_predicted.cpu().detach().numpy()
    # print('labels_gt_np = ' + str(labels_gt_np))
    # print('labels_predicted_np = ' + str(labels_predicted_np))

    # Count FP, FN, TP, and TN
    TPs = []
    FPs = []
    TNs = []
    FNs = []
    precisions = []
    recalls = []
    f1_scores = []
    for ground_truth, probabilitie in zip(ground_truths, probabilities):
        TP, FP, TN, FN = 0, 0, 0, 0
        for gt, pred in zip(ground_truth, probabilitie):
            
            if gt == 1 and pred == 1:  # True positive
                TP += 1
            elif gt == 0 and pred == 1:  # False positive
                FP += 1
            elif gt == 1 and pred == 0:  # False negative
                FN += 1
            elif gt == 0 and pred == 0:  # True negative
                TN += 1
        TPs.append(TP)
        FPs.append(FP)
        FNs.append(FN)
        TNs.append(TN)
        if (TP + FP) == 0  :
            precision = 0
        else:
            precision = TP / (TP + FP)
        precisions.append(precision)
        if (TP + FN) == 0:
            recall = 0
        else:  
            recall = TP / (TP + FN)
        recalls.append(recall)
        if precision == 0 and recall == 0 :
            f1_score = 0
        else:
            f1_score = 2 * (precision*recall)/(precision+recall)
        f1_scores.append(f1_score)

    # print('TP = ' + str(TPs))    
    # print('TN = ' + str(TNs))
    # print('FP = ' + str(FPs))
    # print('FN = ' + str(FNs))
    prec=[]
    reca=[]
    f1_s=[]

    # Compute precision and recall

    for f1_ in f1_scores:
        if f1_ == 0 :
            pass
        else:
            f1_s.append(f1_)
    for reca_ in recalls:
        if reca_ == 0 :
            pass
        else:
            reca.append(reca_)
    for prec_ in precisions:
        if prec_ == 0 :
            pass
        else:
            prec.append(prec_)
    # for i,(pre,rec,f1_sc) in enumerate(zip(prec,reca,f1_s)):
        # print(' Oject ',labs1[i+1],'\n precision ', pre,'\n recall ', rec , '\n f1_sc ',f1_sc)
    
    
    

    return label_res


# label = Call_Md_2d()
# print('probs',label)
