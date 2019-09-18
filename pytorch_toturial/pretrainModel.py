import torchvision.models as models
import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

ResNet18=models.resnet18(pretrained=True)
mnist_data_test = torchvision.datasets.MNIST(root='./ImageData', train = False, transform= transforms.ToTensor(), download=True) # download testing data set

ResNet18.eval()
