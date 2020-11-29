import torch
import torch.nn as nn
import torch.nn.functional as F

#-----------------------------------------------------------------------------------------------------------#
# Model with 5 layer convolutional network
class CNN_5Layer(nn.Module):
    
    
    def __init__(self, number_of_kernels, first_layer_neurons):

        super(CNN_5Layer, self).__init__()
        self.number_of_kernels = number_of_kernels
        
        self.conv1 = nn.Conv2d(3,number_of_kernels,3)
        self.conv2 = nn.Conv2d(number_of_kernels,number_of_kernels,3)
        self.conv3 = nn.Conv2d(number_of_kernels,number_of_kernels,3)
        self.conv4 = nn.Conv2d(number_of_kernels,number_of_kernels,3)
        self.conv5 = nn.Conv2d(number_of_kernels,number_of_kernels,3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(number_of_kernels*7*14,first_layer_neurons) 
        self.fc2 = nn.Linear(first_layer_neurons,10)
        self.fc3 = nn.Linear(10,1)

    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        #print(x.size())
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.size())
        x = self.pool(F.relu(self.conv3(x)))
        #print(x.size())
        x = self.pool(F.relu(self.conv4(x)))
        #print(x.size())
        x = self.pool(F.relu(self.conv5(x)))
        #print(x.size())
        x = x.view(-1, self.number_of_kernels*7*14) # make x an tensor to input into MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze()     
        
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
        x = (F.sigmoid(self.fc3(x))).squeeze()
        
        return x 

        