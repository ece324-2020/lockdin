from NNFullModel import *
from models import *
from lockdin_tools import *
from setcreation import *
from accuracy_calculator import *
from featuremap import *

import argparse
import os

import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
#-----------------------------------------------------------------------------------------------------------# 

def main():
    
    #--------------------- Hyperparameters ---------------------# 
    batch_size = 30
    epochs = 50
    learning_rate = 0.01
    sample_rate = 1
    seed = 10
    
    #--------------------- Data Proccessing ---------------------#
    train_data, valid_data, test_data, overfit_data = setcreation(seed, batch_size)
    
    #--------------------- Model Initialization ---------------------#
    torch.manual_seed(seed)
    NN_model = CNN_2(15, 10)
    optimizer = torch.optim.SGD(NN_model.parameters(),lr=learning_rate)
    loss_function = torch.nn.MSELoss()
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
    
#main()

def extra():
    batch_size = 30
    epochs = 44
    learning_rate = 0.01
    sample_rate = 1
    seed = 10
    
    train_data, valid_data, test_data, overfit_data = setcreation(seed, batch_size)
    
    model = torch.load('saved_models/CNN_2Layer_1.pt')
    #feature_maps1(NN_model)
    #feature_maps2(NN_model)
    

    
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    loss_function = torch.nn.BCELoss()
    acc = accuracy_calculator
    df = decision_function
    
    Full_Model = NNFullModel(model, loss_function, optimizer, acc, decision_function)
    architecture = lockdin_tools(Full_Model)
    architecture.print_incorrect_predictions(test_data)
    

#extra()

def demo():
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize( (0.5,0.5,0.5), (0.5,0.5,0.5) )])
    mydata = torchvision.datasets.ImageFolder('./finaldataset', transform=transform)
    
    for image, label in mydata:
        img = image / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        NN_model = torch.load('saved_models/CNN_2Layer_1.pt')
        output = NN_model(image.unsqueeze(0))
        
        print(output)
        

        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        NN_model.conv2.register_forward_hook(get_activation('conv2'))
        image.unsqueeze_(0).reshape
        output = NN_model(image)
        act = activation['conv2'].squeeze()
        fig, axarr = plt.subplots(5,2)
        for j in range(0,5):
            for idx in range(0,2):
                axarr[j][idx].imshow(act[j*2+idx])
        plt.show()
        
demo()

