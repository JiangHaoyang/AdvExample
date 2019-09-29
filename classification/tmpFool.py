import MyModel
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np

from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

def qFool(net, image, label):
    
    backIsAvailable = True
    
    # set the net in eval mode
    net.eval()
    
    # randomly select the starting point
    pertImage = []
    while True:
        pertImage = torch.randn(image.size()).to(image.device)
        if net(pertImage).argmax() != label:
            break
    # push the pert point just outside the boundary
    pertImage = searchBound(net, image, label, pertImage - image)
    
    # calculate the normal vector of the boundary
    if backIsAvailable:
        out = net(pertImage)
        out[0,label].backward( retain_graph = True )
        normalVector = pertImage.grad.clone()
        normalVector = normalVector / normalVector.norm()
        pass
    else:
        pass
    
    # calculate the step direction
    # how does high dimensional outer product define?
    
    
    
    
    return 0

def searchBound(net, image, label, direction):
    while net(image + direction).argmax() == label:
        direction = direction * 2

    pertNorm = direction.norm()
    while( pertNorm >= 1 ):
        print(pertNorm)
        pertNorm = pertNorm / 2
        if net(image + direction).argmax() != label:
            direction -= direction * pertNorm
        else:
            direction += direction * pertNorm
    return (image + direction)


if __name__ == '__main__':
    
    toPIL=transforms.ToPILImage()
    toTensor=transforms.ToTensor()
    # device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load model
    net = MyModel.LeNet_my(10).to(device)

    # load pretrained parameters
    checkpoint = torch.load('../pretrainModel/ResNet_MNIST.ckpt',map_location = device)
    net.load_state_dict(checkpoint)

    # load MNIST dataset
    test_data = torchvision.datasets.MNIST(root='../ImageData', train=False, transform=toTensor,
                                        download=True)  # download testing data set

    # get one image in the image dataset
    image,label = test_data[0]

    netInput = image.unsqueeze(0).to(device)
    rightLabel = torch.Tensor([label]).long().to(device)

    # show the image
    # Imag = toPIL(image)
    # Imag.show()

    # calculate the perturbation
    grad_sign = qFool(net,netInput,rightLabel)

    # image_perturbated = torch.clamp((image+0.25*grad_sign.cpu()),0,1).to(device)

    # # show the image
    # Imag = toPIL(image)
    # Imag.show()

    # # image_perturbated = torch.Tensor(image_perturbated).to(device)
    # Imag_perturbated=toPIL(image_perturbated.squeeze(0).cpu())
    # Imag_perturbated.show()

    
    # net.eval()

    # print(net(image_perturbated))
    # print(net(image.unsqueeze(0).to(device)))
    # _,predictedLabel = torch.max(net(image_perturbated),1)
    # print(predictedLabel)