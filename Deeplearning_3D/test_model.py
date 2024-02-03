from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics import Precision, Recall, F1Score
from torchmetrics.classification import MulticlassPrecision
import json

try:
    from classes import PointNet, PointCloudData, default_transforms
except ImportError:
    # This happens when running from outside of this file
    from .classes import PointNet, PointCloudData, default_transforms


def testModel(model_path, file_count=200, batch_size=10, metrics_averaging="macro"):  # 200 / 10 by default

    with open('PointCloud_Learning/dataset_filenames_off.json', 'r') as f:
            dataset_filenames = json.load(f)


    # file_count can't be more than number of files. "0" means all files
    test_filenames = dataset_filenames['test_filenames']

    if file_count < 1:
        file_count = len(test_filenames)
    else:
        file_count = min(len(test_filenames), file_count)
    test_filenames = test_filenames[0:file_count]

    print(f"Testing with {file_count} files...")


    # Batch size should not be bigger than total files
    batch_size = min(batch_size, file_count)

    test_ds = PointCloudData(valid=True, filenames=test_filenames, transform=default_transforms)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size)

    classes={}
    labels = ['apple', 'ball', 'banana', 'bell pepper', 'binder', 'bowl', 'calculator',
               'camera', 'cap', 'cell phone', 'cereal box', 'coffee mug', 'comb', 'dry battery',
                'flashlight', 'food bag', 'food box', 'food can', 'food cup', 'food jar',
                'garlic', 'glue stick', 'greens', 'hand towel', 'instant noodles', 'keyboard',
                'kleenex', 'lemon', 'lightbulb', 'lime', 'marker', 'mushroom', 'notebook',
                'onion', 'orange', 'peach', 'pear', 'pitcher', 'plate', 'pliers', 'potato',
                'rubber eraser', 'scissors', 'shampoo', 'soda can', 'sponge', 'stapler', 
                'tomato', 'toothbrush', 'toothpaste', 'water bottle']
    for i,name in enumerate(labels):
        classes[name] = {i}
            
        # When unspecified
    classes['unspecified']={-1}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pointnet = PointNet()
    pointnet.to(device)
    pointnet.load_state_dict(torch.load(model_path))


    pointnet.eval()
    all_preds = []
    all_gt_labels = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            print('Batch [%4d / %4d]' % (i+1, len(test_loader)))
            
            inputs, labels = data['pointcloud'].float(), data['category']
            outputs, __, __ = pointnet(inputs.transpose(1,2))
            _, preds = torch.max(outputs.data, 1)
            all_preds += list(preds.numpy())
            all_gt_labels += list(labels.numpy())



    cm = confusion_matrix(all_gt_labels, all_preds)



    # function from https://deeplizard.com/learn/video/0LhiS6yu2qQ
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    plt.figure(figsize=(8,8), num='Normalized Confusion Matrix')
    plot_confusion_matrix(cm, list(classes.keys()), normalize=True)
    plt.show()



    plt.figure(figsize=(8,8), num='Confusion Matrix')
    plot_confusion_matrix(cm, list(classes.keys()), normalize=False)
    plt.show()


    #--------------------------------------------------------------
    # Metrics -----------------------------------------------------
    #--------------------------------------------------------------

    all_preds_np = []
    all_gt_labels_np = []
    for i, j in zip(all_preds, all_gt_labels):
        all_preds_np.append(i.item())
        all_gt_labels_np.append(j.item())


    tensor_preds = torch.tensor(all_preds_np)
    tensor_gt_labels = torch.tensor(all_gt_labels_np)


    precision = Precision(task="multiclass", average=metrics_averaging, num_classes=5)
    recall = Recall(task="multiclass", average=metrics_averaging, num_classes=5)
    f1_score = F1Score(task="multiclass", num_classes=5)
    multiclass_precision = MulticlassPrecision(num_classes=5, average=None)

    classes_precision = list(multiclass_precision(tensor_preds, tensor_gt_labels))

    print("\n------------------Metrics------------------")
    print("\n" + metrics_averaging.capitalize() + "-Averaging Precision: {:.1f}%".format(float((precision(tensor_preds, tensor_gt_labels)).item() * 100)))
    print(metrics_averaging.capitalize() + "-Averaging Recall: {:.1f}%".format(float((recall(tensor_preds, tensor_gt_labels)).item() * 100)))
    print("F1 Score: {:.1f}%\n".format(float((f1_score(tensor_preds, tensor_gt_labels)).item() * 100)))

    new_classes_precision = []
    for class_precision in  classes_precision:
        new_classes_precision.append(float((class_precision).item()))

    for i, precision in enumerate(new_classes_precision):
        print("Class '{}' Precicion: {:.1f}%\n".format(str(list(classes.keys())[list(classes.values()).index(i)]), (precision * 100)))
        

if __name__ == '__main__':
    testModel(model_path='models/save.pth')

