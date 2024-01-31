import os
import torch
import torch.nn.functional as F
from deeplearning.model import Model


# Create an instance of the model
loaded_model = Model()

# Load the saved state_dict into the model
checkpoint = torch.load("models/checkpoint_t2.pkl")
loaded_model.load_state_dict(checkpoint["model_state_dict"])

# Set the model to evaluation mode (important if using BatchNorm or Dropout)
loaded_model.eval()



def Call_Md(inputs):
    outputs = loaded_model(inputs)
    labels_predicted = loaded_model.forward(inputs)
    # Transform predicted labels into probabilities
    predicted_probabilities = F.softmax(labels_predicted, dim=1).tolist()
    # print(' predicted' + str(predicted_probabilities))
    probabilities = [ []  for i in range(51)]
    # probabilities_dog = [x[0] for x in predicted_probabilities]
    for x in predicted_probabilities:
        for i, probabilitie in enumerate(probabilities):
            
            probabilitie.append(x[i] > 0.95 )

    
    return outputs,probabilities

