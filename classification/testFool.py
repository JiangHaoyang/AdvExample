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

image_perturbated = netInput

for i in range(5):
    grad_sign = foolModel.FGSM(net,image_perturbated,rightLabel)
    image_perturbated = image_perturbated + 0.05 * grad_sign
    net.eval()
    out = net(image_perturbated)
    print(out)
    _,predictedLabel = torch.max(out,1)
    print(predictedLabel)
    print(torch.norm(image_perturbated - netInput, float('inf')))
    print('#####################################')
    pass
