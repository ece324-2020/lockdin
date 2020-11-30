
def feature_maps1():
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
    fig, axarr = plt.subplots(5,8)
    for j in range(0,5):
        for idx in range(0,8):
            axarr[j][idx].imshow(act[j*8+idx])
    plt.show()

def feature_maps1():
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
    act = activation['conv1'].squeeze()
    fig, axarr = plt.subplots(5,8)
    for j in range(0,5):
        for idx in range(0,8):
            axarr[j][idx].imshow(act[j*8+idx])
    plt.show()
