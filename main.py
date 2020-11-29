from NNFullModel import *
from models import *
from lockdin_tools import *
from setcreation import *
from accuracy_calculator import *

import argparse
import os

import torch.nn as nn
import torch.optim as optim

#-----------------------------------------------------------------------------------------------------------# 

def main():
    
    #--------------------- Hyperparameters ---------------------# 
    batch_size = 15
    epochs = 100
    learning_rate = 0.001
    sample_rate = 1
    seed = 10
    
    #--------------------- Data Proccessing ---------------------#
    train_data, valid_data, test_data, overfit_data = setcreation(seed, batch_size)
    
    #--------------------- Model Initialization ---------------------#
    torch.manual_seed(seed)
    NN_model = baseline()
    optimizer = torch.optim.SGD(NN_model.parameters(),lr=learning_rate)
    loss_function = torch.nn.MSELoss()
    acc = accuracy_calculator
    
    Full_Model = NNFullModel(NN_model, loss_function, optimizer, acc)
    
    #--------------------- Running Training ---------------------# 
    architecture = lockdin_tools(Full_Model)
    
    architecture.regular_training(epochs, train_data, valid_data, test_data, sample_rate)
    
    #--------------------- Display Results ---------------------# 
    
    architecture.display_results()
    
    #--------------------- Save Model ---------------------#
    architecture.save_model()
    
main()
