import torch
import numpy as np
import time
import foolbox
import foolbox.distances as distances
from setup import experimentSetup

if __name__ == "__main__":
    [net, testset, trainset, testloader, trainloader, device] = experimentSetup('LeNet','MNIST',pretrain = True)
    net.eval()

    fmodel = foolbox.models.PyTorchModel(net, bounds=(0,1),num_classes=10)
    attack = foolbox.attacks.FGSM(fmodel)

    image,label = testset[0]
    image = image.numpy()
    label = int(label)
    print(label)
    adv = attack(image,3)
    print(adv)