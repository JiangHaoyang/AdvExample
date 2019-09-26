import MyModel
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import foolModel

toPIL=transforms.ToPILImage()
toTensor=transforms.ToTensor()
# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load model
net = MyModel.LeNet_my(10).to(device)

# load pretrained parameters
checkpoint = torch.load('../pretrainModel/ResNet_MNIST.ckpt',map_location = device)
net.load_state_dict(checkpoint)

test_data = torchvision.datasets.MNIST(root='../ImageData', train=False, transform=toTensor,
                                    download=True)  # download testing data set

imageNum = len(test_data)

test_loader = torch.utils.data.DataLoader(dataset = test_data,
                                          batch_size = 100,
                                          shuffle = False)

total = 0
correct = 0

for i, (images,labels) in enumerate(test_loader):

    netInput = torch.autograd.Variable(images,requires_grad = True)

    grad_sign = foolModel.FGSM(net,netInput.to(device),labels.long().to(device))
    image_perturbated = (images + 1*grad_sign.cpu()).numpy()
    image_perturbated = np.clip(image_perturbated,0,1)
    image_perturbated = torch.Tensor(image_perturbated).to(device)

    net.eval()

    outputs = net(image_perturbated)
    _,predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum().item()


    print(i)
    pass

print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

print('finish')