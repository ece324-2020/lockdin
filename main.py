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
    batch_size = 40
    epochs = 100
    learning_rate = 0.1
    sample_rate = 1
    seed = 3
    
    #--------------------- Data Proccessing ---------------------#
    train_data, valid_data, test_data, overfit_data = setcreation(seed, batch_size)
    
    #--------------------- Model Initialization ---------------------#
    torch.manual_seed(seed)
    NN_model = CNN_2Layer(4,30)
    optimizer = torch.optim.SGD(NN_model.parameters(),lr=learning_rate)
    loss_function = torch.nn.MSELoss()
    acc = accuracy_calculator
    
    Full_Model = NNFullModel(NN_model, loss_function, optimizer, acc)
    
    #--------------------- Running Training ---------------------# 
    architecture = lockdin_tools(Full_Model)
    
    architecture.overfit_training(epochs, overfit_data)
    
    #--------------------- Display Results ---------------------# 
    
    architecture.display_results()
    
main()
