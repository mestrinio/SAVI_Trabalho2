#!/usr/bin/env python3


import glob
import json
from sklearn.model_selection import train_test_split
from dataset_random2 import Dataset
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from model import Model
from trainer import Trainer

import torch.nn.functional as F


def main():

    # -----------------------------------------------------------------
    # Hyperparameters initialization
    # -----------------------------------------------------------------
    learning_rate = 0.001
    num_epochs = 20

    # -----------------------------------------------------------------
    # Create model
    # -----------------------------------------------------------------
    model = Model()

    # -----------------------------------------------------------------
    # Prepare Datasets
    # -----------------------------------------------------------------
    with open('dataset_filenames.json', 'r') as f:
        # Reading from json file
        dataset_filenames = json.load(f)

    test_filenames=dataset_filenames['test_filenames']
    # test_filenames = dataset_filenames['test_filenames']
       
    # test_filenames = test_filenames[0:100]

    print('Used ' + str(len(test_filenames)) + ' for testing ')

    test_dataset = Dataset(test_filenames)

    batch_size = len(test_filenames)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # Just for testing the train_loader
    tensor_to_pil_image = transforms.ToPILImage()

    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load the trained model
    checkpoint = torch.load('models/checkpoint_t3.pkl')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()  # we are in testing mode
    batch_losses = []
    for batch_idx, (inputs, labels_gt) in enumerate(test_loader):

        # move tensors to device
        inputs = inputs.to(device)
        labels_gt = labels_gt.to(device)
        print(labels_gt)

        # Get predicted labels
        labels_predicted = model.forward(inputs)


    # Transform predicted labels into probabilities
    predicted_probabilities = F.softmax(labels_predicted, dim=1).tolist()
    # max_pred =max(predicted_probabilities)
    # print(str(max_pred))
    # print(' predicted' + str(predicted_probabilities))
    probabilities = [ []  for i in range(51)]
    # probabilities_dog = [x[0] for x in predicted_probabilities]
    for x in predicted_probabilities:
        max_pred = max(x)
        for i, probabilitie in enumerate(probabilities):
            # print(probabilitie)
            probabilitie.append(x[i] > 0.50 )
    

    # Make a decision using the largest probability
    # predicted_is_dog = [x > 0.5 for x in probabilities_dog]
    

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
        precision = TP / (TP + FP)
        precisions.append(precision)
        recall = TP / (TP + FN)
        recalls.append(recall)
        f1_score = 2 * (precision*recall)/(precision+recall)
        f1_scores.append(f1_score)

    print('TP = ' + str(TPs))    
    print('TN = ' + str(TNs))
    print('FP = ' + str(FPs))
    print('FN = ' + str(FNs))

    # Compute precision and recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision*recall)/(precision+recall)
    


    labels = ['apple', 'ball', 'banana', 'bell_pepper', 'binder', 'bowl', 'calculator',
               'camera', 'cap', 'cell_phone', 'cereal_box', 'coffee_mug', 'comb', 'dry_battery',
                'flashlight', 'food_bag', 'food_box', 'food_can', 'food_cup', 'food_jar',
                'garlic', 'glue_stick', 'greens', 'hand_towel', 'instant_noodles', 'keyboard',
                'kleenex', 'lemon', 'lightbulb', 'lime', 'marker', 'mushroom', 'notebook',
                'onion', 'orange', 'peach', 'pear', 'pitcher', 'plate', 'pliers', 'potato',
                'rubber_eraser', 'scissors', 'shampoo', 'soda_can', 'sponge', 'stapler', 
                'tomato', 'toothbrush', 'toothpaste', 'water_bottle']
    classes={}
    
        
    for i,name in enumerate(labels):
      
        # dic_names= {name}
        classes[i]={"class_index": i,
                        "class_label": name,
                        "metrics": {
                        "precision": 0 ,
                        "recall": 0 ,
                        "f1_score":0 ,
                        "TP": 0,
                        "TN": 0,
                        "FP": 0,
                        "FN": 0
                        }}
        
        # class_['class_index']=i
        # class_['class_label']=name 

    # for i,x in enumerate(precisions):
    #     # print('Precision = ' + str(precisions[i]))
    #     classes['metrics']['precision'] = x
    #     i=i+1

    for i in range(51):
        classes[i]['metrics']['precision']=precisions[i]
        classes[i]['metrics']['recall']=recalls[i]
        classes[i]['metrics']['f1_score']=f1_scores[i]
        classes[i]['metrics']['TP']=TPs[i]
        classes[i]['metrics']['TN']=TNs[i]
        classes[i]['metrics']['FP']=FPs[i]
        classes[i]['metrics']['FN']=FNs[i]



    # for i,_ in enumerate(recalls):
    #     print('Recall = ' + str(recalls[i]))
    #     i=i+1

    # for i,_ in enumerate(f1_scores):
    #     print('F1 score = ' + str(f1_scores[i]))
    #     i=i+1    

 

    # Show image
    # inputs = inputs.cpu().detach()
    # print(inputs)
    for i in range(51):
        print(classes[i])

    fig = plt.figure()
    idx_image = 0
    for row in range(7):
        for col in range(8):
            image_tensor = inputs[idx_image, :, :, :]
            image_pil = tensor_to_pil_image(image_tensor)
            # print('ground_truth is dog = ' + str(ground_truth_is_dog[idx_image]))
            # print('predicted is dog = ' + str(predicted_is_dog[idx_image]))

            ax = fig.add_subplot(7, 8, idx_image+1)
            plt.imshow(image_pil)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

            # text = 'GT '
            
            # if ground_truths[0][idx_image]:
            #     text += 'is dog'
            # else:
            #     text += 'is not dog'

            # text += '\nPred '
            # if predicted_probabilities[0][idx_image]:
            #     text += 'is dog'
            # else:
            #     text += 'is not dog'

            # if ground_truths[0][idx_image] == predicted_probabilities[0][idx_image]:
            #     color = 'green'
            # else:
            #     color = 'red'

            # ax.set_xlabel(text, color=color)
            if idx_image <56:
                idx_image += 1

    plt.show()
    


    # plt.show()
if __name__ == "__main__":
    main()
