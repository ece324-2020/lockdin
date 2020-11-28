from NNFullModel import *
from models import *
from lockdin_tools import *
from dataprocess import *
from accuracy_calculator import *

import argparse
import os

import torch.nn as nn
import torch.optim as optim

#-----------------------------------------------------------------------------------------------------------# 

def main():
    
    #--------------------- Hyperparameters ---------------------# 
    batch_size = 22
    epochs = 100
    learning_rate = 0.05
    sample_rate = 1
    seed = 3
    
    #--------------------- Data Proccessing ---------------------#
    train_data, valid_data, test_data = setcreation(seed, batch_size)
    
    #--------------------- Model Initialization ---------------------#
    NN_model = CNN_2Layer(10,32)
    optimizer = torch.optim.SGD(NN_model.parameters(),lr=learning_rate)
    loss_function = torch.nn.MSELoss()
    acc = accuracy_calculator
    
    Full_Model = NNFullModel(NN_model, loss_function, optimizer, acc)
    
    #--------------------- Running Training ---------------------# 
    architecture = lockdin_tools(Full_Model)
    
    architecture.regular_training(epochs, train_data, valid_data, test_data, sample_rate)
    
    #--------------------- Display Results ---------------------# 
    
    architecture.display_results()
    
main()
