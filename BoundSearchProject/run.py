import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import attackModel
import time
import sys
sys.path.append('..')
import pytorch_cifar.models as models


# device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################################################################################################## select model
# load classification network
# net = models.vgg.VGG('VGG16')             # vgg16 applying on cifar10
# net = models.resnet.ResNet18()            # resnet18 applying on cifar10
net = models.MyModel.Model_1()            # modified LeNet applying on MNIST
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

############################################################################################################## load selected model
# load the pretrian model
# checkpoint = torch.load('../pretrainModel/Vgg16_Cifar10_ckpt.pth')
# checkpoint = torch.load('../pretrainModel/ResNet18_Cifar10_ckpt.pth')
checkpoint = torch.load('../pretrainModel/LeNet_MNIST_ckpt.pth')
net.load_state_dict(checkpoint['net'])
# set the network in eval mode
net = net.eval().to(device)
print('the network achieve', checkpoint['acc'],'in testing set')

CIFAR_transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
# MNIST_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
MNIST_transform = transforms.Compose([transforms.ToTensor()])

############################################################################################################## select data
# testset = torchvision.datasets.CIFAR10(root='../ImageData', train=False, download=True, transform=CIFAR_transform)
testset = torchvision.datasets.MNIST(root='../ImageData', train=False, download=True, transform=MNIST_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

numOfCorrect = 0
numOfSuccessFakeAttack = 0
numOfSuccessRealAttack = 0
ImageNum = len(testset)
norms = torch.zeros([ImageNum])
i = 0
startTime = time.time()
for imageIndex, (image,label) in enumerate(testloader):
    if ((imageIndex+1) % 10 != 0):
        continue
    label = torch.Tensor([label]).long().to(device)
    image = image.to(device)
    if(label == net(image).argmax()):
        numOfCorrect += 1
        ############################################################################################# select attack model
        # attack the model here
        pertImage = attackModel.BoundSearchAttack(net,image,label)
        if(label != net(pertImage).argmax()):
            numOfSuccessFakeAttack += 1
        # clip image to valid
        validPertImage = pertImage.clone()
        validPertImage[pertImage<0] = 0
        validPertImage[pertImage>1] = 1

        if(label != net(validPertImage).argmax()):
            numOfSuccessRealAttack += 1
            norms[i] = (image - pertImage).norm()
            i += 1
    if ((imageIndex+1) % 100 == 0):
        print(imageIndex+1,'/',len(testset))
    pass
endTime = time.time()
print(endTime - startTime)
print('numOfCorrect',numOfCorrect)
print('numOfSuccessFakeAttack',numOfSuccessFakeAttack)
print('numOfSuccessRealAttack',numOfSuccessRealAttack)
print('average norm is',(norms[0:i]).mean())