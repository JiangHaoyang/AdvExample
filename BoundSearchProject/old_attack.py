import numpy as np
import torch
import time
from setup import experimentSetup
import attackModel

def randImage(label):
    global indexByLabel
    global testset
    tmp = np.linspace(0,10,10,endpoint=False,dtype=np.int)
    randLabel = np.random.choice(tmp)
    while randLabel == label:
        randLabel = np.random.choice(tmp)
        pass
    randImageIndex = np.random.randint(0,849)
    startingImage = testset[indexByLabel[randLabel,randImageIndex].long()][0].unsqueeze(0).to(device)
    return startingImage

[net, testset, trainset, testloader, trainloader, device] = experimentSetup('LeNet','MNIST',pretrain = True)
net.eval()

numOfCorrect = 0
numOfSuccessFakeAttack = 0
numOfSuccessRealAttack = 0
ImageNum = len(testset)
norms = torch.zeros([ImageNum])
startTime = time.time()

# locate certain image with its label
indexByLabel = torch.zeros([10,850])
k = np.zeros([10])
for i in range(len(testset)):
    image,label = testset[i]
    if (k[label] >= 850): continue
    indexByLabel[label,int(k[label])] = i
    k[label] += 1

confMatirx = torch.zeros([10,10],device=device)
aveNorm = torch.zeros([10,10],device=device)


i = 0
for imageIndex, (image,label) in enumerate(testloader):
    if ((imageIndex+1) % 10 != 0):
        continue
    label = torch.Tensor([label]).long().to(device)
    image = image.to(device)
    tmp = net(image).argmax()
    if(label == tmp):
        numOfCorrect += 1
        #
        startingPoint = False
        if startingPoint == True:
            # select a random image whose label is not this one
            startingImage = randImage(label)
        ############################################################################################# select attack model
        # attack the model here
        pertImage = attackModel.IFGSM(net,image,label)
        tmp = net(pertImage).argmax()
        if(label != tmp):
            confMatirx[label, tmp] += 1
            numOfSuccessFakeAttack += 1
            norms[i] = (image - pertImage).norm()
            aveNorm[label,tmp] += norms[i]
            i += 1
        # clip image to valid
        validPertImage = pertImage.clone()
        validPertImage[pertImage < 0] = 0
        validPertImage[pertImage > 1] = 1

        tmp = net(validPertImage).argmax()
        if(label != tmp):
            # confMatirx[label,tmp] += 1
            numOfSuccessRealAttack += 1
            # norms[i] = (image - pertImage).norm()
            # aveNorm[label,tmp] += norms[i]
            # i += 1
    if ((imageIndex+1) % 100 == 0):
        print(imageIndex+1,'/',len(testset))
    pass
aveNorm = aveNorm/confMatirx
endTime = time.time()
print(endTime - startTime)
print('numOfCorrect',numOfCorrect)
print('numOfSuccessFakeAttack',numOfSuccessFakeAttack)
print('numOfSuccessRealAttack',numOfSuccessRealAttack)
print('average norm is',(norms[0:i]).mean())
print(confMatirx)
print(aveNorm)