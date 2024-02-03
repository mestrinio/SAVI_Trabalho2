import os
import torch
import torch.nn.functional as F
from deeplearning.model import Model
from Deeplearning_3D.classes import PointNet, PointCloudData, default_transforms
from torch.utils.data import DataLoader


model_path = 'models/checkpoint_3D.pth'

test_ds = PointCloudData(valid=True, filenames=test_filenames, transform=default_transforms)
test_loader = DataLoader(dataset=test_ds, batch_size=batch_size)
# Create an instance of the model
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