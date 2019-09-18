import torchvision.models as models
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable

#download model
ResNet18 = models.resnet18(pretrained=True)
ResNet18.eval()

#define how an PIL image transforms into the network input
inputTransformer=transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

#download data
imageData_train = torchvision.datasets.ImageNet(root='./ImageData/imageNet', train = True, transform = inputTransformer, download=True) #download training data set
imageData_test = torchvision.datasets.ImageNet(root='./ImageData/imageNet', train = False, transform = inputTransformer, download=True) #download testing data set

# image,label = imageData_train[0]
# predict = ResNet18.forward(Variable(image[None,:,:,:],requires_grad = True))

# print(predict.shape)

#print(output)