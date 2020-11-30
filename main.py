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
    epochs = 1
    learning_rate = 0.01
    sample_rate = 1
    seed = 10
    
    #--------------------- Data Proccessing ---------------------#
    train_data, valid_data, test_data, overfit_data = setcreation(seed, batch_size)
    
    #--------------------- Model Initialization ---------------------#
    torch.manual_seed(seed)
    NN_model = CNN_2(40, 10)
    optimizer = torch.optim.SGD(NN_model.parameters(),lr=learning_rate)
    loss_function = torch.nn.BCELoss()
    acc = accuracy_calculator
    df = decision_function
    
    Full_Model = NNFullModel(NN_model, loss_function, optimizer, acc, decision_function)
    
    #--------------------- Running Training ---------------------# 
    architecture = lockdin_tools(Full_Model)
    
    architecture.regular_training(epochs, train_data, valid_data, test_data, sample_rate)
    
    #--------------------- Display Results ---------------------# 
    
    architecture.display_results()
    architecture.confusion_matrix(test_data)
    
    #--------------------- Save Model ---------------------#
    architecture.save_model()
    # baseline, 100 epochs, 0.001 learning rate, batch size 15
    # CNN1(40,25), 100 epochs, batch size 15
    
main()
