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
    return (0.25*grad_sign+image)


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

    return pertImage


def qFool(net, image, label, startingPoint = None):
    
    toPIL=transforms.ToPILImage()

    backwardIsAvailable = True

    # set the net in eval mode
    net.eval()
    randomStart = True
    if randomStart:
        # randomly select the starting point
        pertImage = []
        while True:
            pertImage = torch.randn(image.size()).to(image.device) # sample from standard normal distribution
            pertImage = pertImage/5+0.5 # 0.5 mean and 1/5 variance, make sure the pert image is valid
            if net(pertImage).argmax() != label:
                break
    else:
        # generate starting point using linear conbination of existing data
        pertImage = startingPoint
        pass
    # push the pert point just outside the boundary
    pertImage = torch.autograd.Variable(searchBound(net, image, label, pertImage - image),requires_grad = True)
    iter = 0
    
    while (iter < 100) & ((image - pertImage).norm().cpu().detach().numpy() >= 1.8 ):
        
        # calculate the normal vector of the boundary
        if backwardIsAvailable:
            out = net(pertImage)
            out[0,label].backward( retain_graph = True )
            normalVector = pertImage.grad.clone()
            normalVector = normalVector / normalVector.norm()
            pass
        else:
            print('else is empty')
            pass
        
        # calculate the step direction
        directionToOriImage = image - pertImage
        dirProjOnNormal = normalVector * (normalVector * directionToOriImage).sum()
        dirOfStep = directionToOriImage - dirProjOnNormal
        pertImage = pertImage + 0.25*dirOfStep
        pertImage = torch.autograd.Variable(searchBound(net, image, label, pertImage - image),requires_grad = True)
        iter += 1
    print(iter)
    return pertImage


def searchBound(net, image, label, direction):
    while net(image + direction).argmax() == label:
        direction = direction * 2

    oriDirection = direction.clone()
    pertNorm = 1
    while (oriDirection.norm() * pertNorm >= 1) :
        pertNorm = pertNorm / 2
        if net(image + direction).argmax() != label:
            direction -= (oriDirection * pertNorm)
        else:
            direction += (oriDirection * pertNorm)
    if net(image + direction).argmax() == label:
        direction += (oriDirection * pertNorm)

    pertImage = image + direction
    # clip to valid
    pertImage[pertImage<0] = 0
    pertImage[pertImage>1] = 1

    return pertImage


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

    i = 30
    rand = np.random.randint(0, len(test_data),[i,1])
    coefficient = np.random.rand(i)
    coefficient = coefficient / sum(coefficient)
    startingPoint = torch.zeros(image.size()).to(device)
    for k in range(i):
        startingPoint += test_data[int(rand[k])][0] * coefficient[k]
        pass

    # calculate the perturbation
    pertImage = qFool(net,netInput,rightLabel,startingPoint)

    print(net(netInput).argmax())
    print('########################')
    print(net(pertImage).argmax())
    print('########################')
    
    # set image to valid
    validPertImage = pertImage.clone()
    validPertImage[pertImage<0] = 0
    validPertImage[pertImage>1] = 1
    print(net(validPertImage).argmax())
    print('########################')

    print((validPertImage-netInput).norm())

    # show the pert image
    PILPertImage = validPertImage.squeeze(0).cpu()
    PILPertImage = toPIL(PILPertImage)
    PILPertImage.show()
    oriImage = toPIL(image)
    oriImage.show()
