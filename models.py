import torch
import torch.nn as nn
import torch.nn.functional as F

#-----------------------------------------------------------------------------------------------------------#
class CNN_1(nn.Module):
    
    
    def __init__(self, number_of_kernels, first_layer_neurons):

        super(CNN_1, self).__init__()
        self.number_of_kernels = number_of_kernels
        
        self.conv1 = nn.Conv2d(3,number_of_kernels,3)
        self.conv2 = nn.Conv2d(number_of_kernels,number_of_kernels,3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(number_of_kernels*71*128,first_layer_neurons) 
        self.fc2 = nn.Linear(first_layer_neurons,1)

    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        #print(x.size())
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.size())
        x = x.view(-1, self.number_of_kernels*71*128) # make x an tensor to input into MLP
        x = F.relu(self.fc1(x))
        x = (F.sigmoid(self.fc2(x))).squeeze()
        
        return x 

#-----------------------------------------------------------------------------------------------------------#
class CNN_2(nn.Module):
    
    
    def __init__(self, number_of_kernels, first_layer_neurons):

        super(CNN_2, self).__init__()
        self.number_of_kernels = number_of_kernels
        
        self.conv1 = nn.Conv2d(3,number_of_kernels,3)
        self.conv2 = nn.Conv2d(number_of_kernels,number_of_kernels,3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(number_of_kernels*34*63,first_layer_neurons) 
        self.fc2 = nn.Linear(first_layer_neurons,1)

    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        #print(x.size())
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.size())
        x = x.view(-1, self.number_of_kernels*34*63) # make x an tensor to input into MLP
        x = F.relu(self.fc1(x))
        x = (F.sigmoid(self.fc2(x))).squeeze()
        
        return x 

#-----------------------------------------------------------------------------------------------------------#

class baseline(nn.Module):
    
    def __init__(self):
        
        super(baseline, self).__init__()
        
        self.fc1 = nn.Linear(145*260*3, 100)
        self.fc2 = nn.Linear(100, 25)
        self.fc3 = nn.Linear(25, 1)
        
    def forward(self, x):
        
        #print(x.shape)
        # Make the picture into a single tensor
        x = x.view(-1, 145*260*3)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = (F.sigmoid(self.fc3(x))).squeeze()
        
        return x 

        