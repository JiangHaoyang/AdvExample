import MyModel
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np

from torch.autograd import Variable
import copy
from torch.autograd.gradcheck import zero_gradients


def FGSM(net, image, label):

    #set the net in eval mode
    net.eval()
    
    #copy image as valid input
    x = image.clone()
    x = torch.autograd.Variable(x,requires_grad=True)
    y = label
    #define loss
    loss_fn = nn.CrossEntropyLoss()

    scores = net(x)
    loss = loss_fn(scores,y)
    loss.backward()
    # grad_sign = x.grad/torch.max(x)
    grad_sign = x.grad.sign()
    return grad_sign


def DeepFool(net, oriImage, label):

    # set the net in eval mode
    net.eval()

    #copy image as valid input
    image = oriImage.clone()
    pertImage = torch.autograd.Variable( image , requires_grad = True )

    # pass forward
    out = net( pertImage )

    sortedOut = (-1*out).argsort() # sort in invert order

    classNum = out.numel()
    iter = 0

    while (out.argmax() == label) & (iter < 50):
        
        dist = np.inf
        # backward to calculate the gradient
        out[0,label].backward(retain_graph=True)
        gradient_label = pertImage.grad.clone()
        pertImage.grad.zero_()

        for k in range(1,classNum):

            ####################### calculate the perturbation for image to class 'sortedOut[0,k]'

            # backward to calculate the gradient
            out[0,sortedOut[0,k]].backward(retain_graph=True)
            gradient_k = pertImage.grad.clone()
            pertImage.grad.data.zero_()

            # different of two gradient
            gradientPrime_k = gradient_k - gradient_label

            # pert = abs(out[0,sortedOut[0,k]] - out[0,sortedOut[0,label]]) / gradientPrime_k.norm() * gradientPrime_k
            dist_k = abs(out[0,sortedOut[0,k]] - out[0,label]) / gradientPrime_k.norm()

            if dist_k < dist:
                dist = dist_k.clone()
                gradient = gradientPrime_k.clone()
        
        pert = (dist+1e-4) * gradient / gradient.norm()
        pertImage = torch.autograd.Variable( pertImage + pert * 1.02 , requires_grad = True )
        iter += 1
        out = net( pertImage )
        sortedOut = (-1*out).argsort() # sort in invert order

    return (pertImage - image)


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
    grad_sign = FGSM(net,netInput,rightLabel)

    image_perturbated = torch.clamp((image+0.25*grad_sign.cpu()),0,1).to(device)

    # show the image
    Imag = toPIL(image)
    Imag.show()

    # image_perturbated = torch.Tensor(image_perturbated).to(device)
    Imag_perturbated=toPIL(image_perturbated.squeeze(0).cpu())
    Imag_perturbated.show()

    
    net.eval()

    print(net(image_perturbated))
    print(net(image.unsqueeze(0).to(device)))
    _,predictedLabel = torch.max(net(image_perturbated),1)
    print(predictedLabel)