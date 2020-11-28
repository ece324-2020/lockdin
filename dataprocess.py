import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

class AdultDataset(data.Dataset):

    def __init__(self, X, y):
    
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = (self.X)[index]
        y = (self.y)[index]
        
        return X,y
    
def setcreation(seed, batch_size):
    
    # Define Transformation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Import Data
    mydata = torchvision.datasets.ImageFolder('./training_v2', transform=transform)
    
    #Extract Data
    images = []
    labels = []
    for image, label in mydata:
        images.append(image.numpy())
        labels.append(label)
        
    # One Hot Encode Labels
    oneh_encoder = OneHotEncoder()
    labels = np.array(labels)
    labels = oneh_encoder.fit_transform(labels.reshape(-1,1)).toarray()
    
    # Train Validation Split
    images = np.array(images)
    imgs_train, imgs_valid, labels_train, labels_valid = train_test_split(images, labels, test_size=0.3, random_state=seed)
    
    # Validation Test Split
    imgs_test, imgs_valid, labels_test, labels_valid = train_test_split(imgs_valid, labels_valid, test_size=0.4, random_state=seed)
    
    # Set Data Type
    imgs_train = imgs_train.astype(np.float32)
    imgs_valid = imgs_valid.astype(np.float32)
    labels_train = labels_train.astype(np.float32)
    labels_valid = labels_valid.astype(np.float32)
    imgs_test = imgs_train.astype(np.float32)
    labels_test = labels_train.astype(np.float32)
    
    
    
    # Create dataset in PyTorch
    Tdata = AdultDataset(imgs_train,labels_train)
    Vdata = AdultDataset(imgs_valid,labels_valid)
    TSTdata = AdultDataset(imgs_test, labels_test)
    
    # Create Batches
    train_data = DataLoader(Tdata, batch_size = batch_size, shuffle = "True")
    valid_data = DataLoader(Vdata, batch_size = batch_size)
    test_data = DataLoader(TSTdata, batch_size = batch_size)
    
    return train_data, valid_data, test_data