import torch
import torch.nn as nn
import torch.nn.functional as F

#-----------------------------------------------------------------------------------------------------------#
# Model with 2 layer convolutional network
class CNN_2Layer(nn.Module):
    
    
    def __init__(self, number_of_kernels, first_layer_neurons):

        super(CNN_2Layer, self).__init__()
        self.number_of_kernels = number_of_kernels
        
        self.conv1 = nn.Conv2d(3,number_of_kernels,5)
        self.conv2 = nn.Conv2d(number_of_kernels,number_of_kernels,5)
        self.pool = nn.MaxPool2d(5,5)
        self.fc1 = nn.Linear(number_of_kernels*10*19,first_layer_neurons) 
        self.fc2 = nn.Linear(first_layer_neurons,1)

    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.number_of_kernels*10*19) # make x an tensor to input into MLP
        x = F.relu(self.fc1(x))
        x = self.fc2(x).squeeze()      
        
        return x 


#-----------------------------------------------------------------------------------------------------------#

class baseline(nn.Module):
    
    def __init__(self):
        
        super(baseline, self).__init__()
        
        self.fc1 = nn.Linear(520*290*3, 100)
        self.fc2 = nn.Linear(100, 25)
        self.fc3 = nn.Linear(25, 1)
        
    def forward(self, x):
        
        # Make the picture into a single tensor
        x = x.view(-1, 520*290*3)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        
        return x 

        