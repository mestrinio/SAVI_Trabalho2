import os
import torch
import torch.nn as nn
from deeplearning.model import Model


# Create an instance of the model
loaded_model = Model()

# Load the saved state_dict into the model
loaded_model.load_state_dict(torch.load('saved_model.pth'))

# Set the model to evaluation mode (important if using BatchNorm or Dropout)
loaded_model.eval()

def Call_Md(inputs):
    outputs = loaded_model(inputs)

    return outputs

