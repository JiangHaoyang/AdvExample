import torch
import numpy as np
import time
import foolbox
import foolbox.distances as distances
from setup import experimentSetup

def dist(pert,L = 'inf'):
    pert = pert.reshape(pert.size)
    if L == 'inf':
        return np.linalg.norm(pert,np.inf)
    if L == '2':
        return np.linalg.norm(pert)
    pass

if __name__ == "__main__":
    [net, testset, trainset, testloader, trainloader, device] = experimentSetup('LeNet','MNIST',pretrain = True)
    net.eval()

    fmodel = foolbox.models.PyTorchModel(net, bounds=(0,1),num_classes=10)
    attack = foolbox.attacks.DeepFoolAttack(fmodel) ############# define attack

    total = 0
    fool = 0
    fail = 0
    norm_mean = 0
    for batchIndex, (images,labels) in enumerate(testloader):
        if (batchIndex+1) % 10 != 0:
            continue
        print(batchIndex+1,'/',len(testloader))
        advImages = torch.zeros(images.size(),device=device)
        for i in range(images.size(0)):
            image = images[i,:,:,:].numpy()
            label = int(labels[i].numpy())

            adversarial = attack(image, label, steps=5, subsample=3) ############# attack parameter
            total += 1
            fool += 1
            if adversarial is None:
                fail += 1
                fool -= 1
                continue
            else:
                pert = adversarial - image
                norm_mean = (norm_mean*(fool-1) + dist(pert,'2')) / fool ############# define distance
    print('fool rate =',fool/total)
    print('average norm =',norm_mean)
    print(total)
    print(fool)
    print(fail)