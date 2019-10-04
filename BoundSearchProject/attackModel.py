import torch
import torch.nn as nn
import numpy as np


def FGSM(net, image, label):

    #set the net in eval mode
    net.eval()
    
    # set eps
    eps = 0.25

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
    return (eps*grad_sign+image)


def DeepFool(net, oriImage, label):

    # set the net in eval mode
    net.eval()

    # set max iter
    maxIter = 50
    # set eps
    eps = 1.8
    # set overshoot
    overshoot = 0.02
    #copy image as valid input
    image = oriImage.clone()
    totalPert = torch.zeros(image.size()).to(oriImage.device)
    pertImage = torch.autograd.Variable( (image + totalPert) , requires_grad = True )
    # pass forward
    out = net( pertImage )

    sortedOut = (-1*out).argsort() # sort in invert order

    classNum = out.numel()
    iter = 0

    while (out.argmax() == label) & (iter < maxIter):
        
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
        totalPert = totalPert + pert
        pertImage = torch.autograd.Variable( image + totalPert*(1+overshoot) , requires_grad = True )
        iter += 1
        out = net( pertImage )
        sortedOut = (-1*out).argsort() # sort in invert order

    return pertImage


def BoundSearchAttack(net, image, label, startingPoint = None):

    backwardIsAvailable = True

    # set max iter
    maxIter = 100
    # set eps
    eps = 1.8

    # set the net in eval mode
    net.eval()
    randomStart = True
    if randomStart:
        # randomly select the starting point
        pertImage = torch.zeros(image.size()).to(image.device)
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
    while (iter < maxIter) & ((image - pertImage).norm().cpu().detach().numpy() >= eps ):
        
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
        pertImage = pertImage + 0.3*dirOfStep
        pertImage = torch.autograd.Variable(searchBound(net, image, label, pertImage - image),requires_grad = True)
        iter += 1
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