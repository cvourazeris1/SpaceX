import torch.nn as nn
import torch.nn.functional as F
import torch

from neural_network import evaluate_model


# Create a Model Class that inherits nn.Module
class Model(nn.Module):

  # input layer (4 features of the flower) --> 4
  # hidden layer 1 (number of neurons) --> 16 
  # hidden layer 2 (number of neurons) --> 16
  # output (3 classes of iris flowers) 3

  def __init__(self, in_features=4, h1=16, h2=16, out_features=3): # initializes parameters of network

    super().__init__()                      # delegates init call to parent class nn.Module

    self.fc1 = nn.Linear(in_features, h1)   # creates fully connected linear layer 
    self.fc2 = nn.Linear(h1, h2)            # creates fully connected linear layer
    self.out = nn.Linear(h2, out_features)  # creates output layer

  def forward(self, x):                     # defines computations performed and flow of data
    x = F.relu(self.fc1(x))                 # relu activation on 1st layer
    x = F.relu(self.fc2(x))                 # relu activation on 2nd layer
    x = F.softmax(self.out(x))              # softmax on final layer to get probabilities 
    return x


def create_model():
    model = Model()
    return model


class ModelEnsemble():

  def __init__(self, models_list, test_dataloader): # initilize the model ensemble
    self.models_list = models_list                  # define the list of models to be used
    self.test_dataloader = test_dataloader          # define the test_dataloader
    self.run_inference()
    self.majority_vote()

  def run_inference(self):                          # get result from each model
    outputs_list = []                               # create list for all model outputs

    for model in self.models_list:                                                      # loop through list of models
      _, _, _, labels, outputs = evaluate_model.test_model(model, self.test_dataloader) # get outputs from a model
      outputs_list.append(outputs.flatten(start_dim=0, end_dim=1)) # go from 3d to 2d by flattening dimension 1

    self.model_outputs = outputs_list # save model outputs
    self.labels = labels.flatten()    # save labels


  def majority_vote(self):
    model_output_sum = sum(self.model_outputs)              # sum all of the model outputs
    ensemble_output = torch.argmax(model_output_sum, dim=1) # find the label of the highest probability
    self.ensemble_output = ensemble_output                  # define that label as the output
