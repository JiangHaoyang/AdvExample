import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import sys
sys.path.append('..')
import Models

# device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################################################################################################## select model
# load classification network
# net = Models.vgg.VGG('VGG16')             # vgg16 applying on cifar10
# net = Models.resnet.ResNet18()            # resnet18 applying on cifar10
net = Models.MyModel.Model_1()            # modified LeNet applying on MNIST

############################################################################################################## load selected model

# load the pretrian model
# checkpoint = torch.load('../pretrainModel/Vgg16_Cifar10_ckpt.pth')
# checkpoint = torch.load('../pretrainModel/ResNet18_Cifar10_ckpt.pth')
checkpoint = torch.load('../pretrainModel/LeNet_MNIST_ckpt.pth')

net.load_state_dict(checkpoint['net'])
# set the network in eval mode
net = net.eval().to(device)
print('the network achieve', checkpoint['acc'],'in testing set')

############################################################################################################## select transforms

# CIFAR_transform = transforms.Compose([transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
# MNIST_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
MNIST_transform = transforms.Compose([transforms.ToTensor()])

############################################################################################################## select data
# testset = torchvision.datasets.CIFAR10(root='../ImageData', train=False, download=True, transform=CIFAR_transform)
testset = torchvision.datasets.MNIST(root='../ImageData', train=False, download=True, transform=MNIST_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
######################################################################################################################################


ToPIL = torchvision.transforms.ToPILImage()

maxIndex = np.zeros([10], dtype=np.int)
indexByLabel = np.zeros([10,2000], dtype=np.int)
for imageIndex, (image,label) in enumerate(testloader):
    indexByLabel[label,maxIndex[label]] = imageIndex
    maxIndex[label] += 1

def synthesisImage(label = 0):
    coeffi = torch.rand([maxIndex[label]])
    coeffi = coeffi/coeffi.sum()
    image = torch.zeros(testset[0][0].size())
    for i in range(maxIndex[label]):
        image += coeffi[i] * testset[indexByLabel[label,i]][0]
    return image

def statistics():
    net.eval()
    confusionMatrix = np.zeros([10,10])
    for i in range(10):
        print(i)
        maxiter = 100
        iterNum = 0
        while iterNum < maxiter :
            iterNum += 1
            image = synthesisImage(i)
            image = image.unsqueeze(0).to(device)
            predict = net(image).argmax().cpu()
            confusionMatrix[i,predict] += 1
            if( (iterNum+1) % 10 == 0):
                print(iterNum+1,'/',maxiter)
    print(confusionMatrix)

def testOneImage(label = 1):
    image = synthesisImage(label)
    ToPIL(image).save('example.png',quality = 95)
    image = image.unsqueeze(0).to(device)
    print(net(image.to(device)).argmax())

if __name__ == "__main__":
    print(maxIndex)
    # statistics()
    testOneImage(label = 9)