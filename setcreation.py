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
    transform = transforms.Compose([transforms.ToTensor()])
    #transforms.Normalize((106.13396973715938, 98.70596573828463, 98.58953582911343), (67.56906106237096, 70.59285965856779, 66.91254745319124))

    # Import Data
    mydata = torchvision.datasets.ImageFolder('./new', transform=transform)

    #Extract Data
    images = []
    labels = []
    mean1 = [0,0,0,0]
    std1 = [0,0,0,0]
    count =0
    for image, label in mydata:
        images.append(image.numpy())
        labels.append(label)
        for a in range(0,4):
            mean1[a] += np.mean(data[:,:,a])
            std1[a] += np.std(data[:,:,a])
        count += 1
    for i in range(0,4):
        mean1[i] = mean1[i] / count
        std1[i] = std1[i] / count

    print(images[1])
    # Train Validation Split
    images = np.array(images)
    labels = np.array(labels)
    imgs_train, imgs_valid, labels_train, labels_valid = train_test_split(images, labels, test_size=0.3, random_state=seed)

    # Validation Test Split
    imgs_test, imgs_valid, labels_test, labels_valid = train_test_split(imgs_valid, labels_valid, test_size=0.4, random_state=seed)

    # Set Data Type
    imgs_train = imgs_train.astype(np.float32)
    imgs_valid = imgs_valid.astype(np.float32)
    labels_train = labels_train.astype(np.float32)
    labels_valid = labels_valid.astype(np.float32)
    imgs_test = imgs_test.astype(np.float32)
    labels_test = labels_test.astype(np.float32)



    # Create dataset in PyTorch
    Tdata = AdultDataset(imgs_train,labels_train)
    Vdata = AdultDataset(imgs_valid,labels_valid)
    TSTdata = AdultDataset(imgs_test, labels_test)

    # Create Batches
    train_data = DataLoader(Tdata, batch_size = batch_size, shuffle = "True")
    valid_data = DataLoader(Vdata, batch_size = batch_size)
    test_data = DataLoader(TSTdata, batch_size = batch_size)

    return train_data, valid_data, test_data, valid_data
