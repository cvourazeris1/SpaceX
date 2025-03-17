import torch.nn as nn
import torch.nn.functional as F
import torch

# Create a Model Class that inherits nn.Module
class Model(nn.Module):
  # Input layer (4 features of the flower) -->
  # Hidden Layer1 (number of neurons) -->
  # H2 (n) -->
  # output (3 classes of iris flowers)
  def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
    super().__init__() # instantiate our nn.Module
    self.fc1 = nn.Linear(in_features, h1)
    self.fc2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.out(x)

    return x

def create_model():
    model = Model()
    return model

class ModelEnsemble():
   
  def __init__(self, models_list, test_dataloader):
    self.models_list = models_list

  def run_inference(self):

    predictions_list = []


    with torch.no_grad():
      for data in self.test_dataloader:
        features, labels = data
        for model in self.models_list:
          outputs = model(features)
          predicted = torch.argmax(outputs, dim=1)
   
   


