
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def feature_maps1(NN_model):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.ImageFolder('./finalDataset', transform=transform)
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    NN_model.conv1.register_forward_hook(get_activation('conv1'))
    data, _ = dataset[0]
    data.unsqueeze_(0).reshape
    output = NN_model(data)
    act = activation['conv1'].squeeze()
    fig, axarr = plt.subplots(5,3)
    for j in range(0,5):
        for idx in range(0,3):
            axarr[j][idx].imshow(act[j*3+idx])
    plt.show()

def feature_maps2(NN_model):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.ImageFolder('./finalDataset', transform=transform)
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    NN_model.conv2.register_forward_hook(get_activation('conv2'))
    data, _ = dataset[0]
    data.unsqueeze_(0).reshape
    output = NN_model(data)
    act = activation['conv2'].squeeze()
    fig, axarr = plt.subplots(5,2)
    for j in range(0,5):
        for idx in range(0,2):
            axarr[j][idx].imshow(act[j*2+idx])
    plt.show()
