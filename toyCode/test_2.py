import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.0005

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

# load MNIST dataset
train_data = torchvision.datasets.CIFAR10(root='../ImageData', train=True, transform=transform,
                                         download=True)  # download training data set
test_data = torchvision.datasets.CIFAR10(root='../ImageData', train=False, transform=transform,
                                        download=True)  # download testing data set