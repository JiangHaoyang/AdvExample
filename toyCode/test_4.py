import torch
import foolbox
import numpy as np
import torchvision
import torchvision.transforms as transforms
import numpy as np
import attackModel
import time
import sys
sys.path.append('..')
import Models
from PIL import Image

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

# device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

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

# CIFAR_transform = transforms.Compose([transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
# MNIST_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
MNIST_transform = transforms.Compose([transforms.ToTensor()])

ArrayToPIL = transforms.Compose([transforms.ToTensor(),transforms.ToPILImage()])
TensorToPIL = transforms.ToPILImage()
############################################################################################################## select data
# testset = torchvision.datasets.CIFAR10(root='../ImageData', train=False, download=True, transform=CIFAR_transform)
testset = torchvision.datasets.MNIST(root='../ImageData', train=False, download=True, transform=MNIST_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

image, label = testset[555]
im = TensorToPIL(image)

image = image.numpy()

fmodel = foolbox.models.PyTorchModel(net, bounds=(0,1),num_classes=10)

print('label',label)
print('predicted class',np.argmax(fmodel.predictions(image)))

attack = foolbox.attacks.ProjectedGradientDescent(fmodel)
adversarial = attack(image, label)
imadv = TensorToPIL(torch.from_numpy(adversarial))
print('adversarial class', np.argmax(fmodel.predictions(adversarial)))

pert = (adversarial-image).reshape([image.size])*255
print(np.linalg.norm(pert,np.inf))

im.save('ori.png')
imadv.save('adv.png')
