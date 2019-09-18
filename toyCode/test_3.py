import torch
import torchvision
import torchvision.transforms as transforms
from ..classification import MyModel
import torch.nn as nn

# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.0005

# load MNIST dataset
train_data = torchvision.datasets.KMNIST(root='../ImageData', train=True, transform=transforms.ToTensor(),
                                        download=True)  # download training data set
test_data = torchvision.datasets.KMNIST(root='../ImageData', train=False, transform=transforms.ToTensor(),
                                       download=True)  # download testing data set

# define dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batch_size,
                                          shuffle=False)

# load model
net = MyModel.LeNet_my(num_classes).to(device)