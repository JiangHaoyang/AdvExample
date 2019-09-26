import MyModel
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np


def FGSM(net, image, label):

    #set the net in eval mode
    net.eval()
    
    #copy image as valid input
    x = torch.autograd.Variable(image,requires_grad=True)
    y = label
    #define loss
    loss_fn = nn.CrossEntropyLoss()

    scores = net(x)
    loss = loss_fn(scores,y)
    loss.backward()
    grad_sign = x.grad/torch.max(x)
    #grad_sign = x.grad.sign()
    return grad_sign




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

    # show the image
    # Imag = toPIL(image)
    # Imag.show()

    # calculate the perturbation
    grad_sign = FGSM(net,netInput,rightLabel)

    image_perturbated = (image+0.25*grad_sign.cpu()).numpy()
    image_perturbated = np.clip(image_perturbated,0,1)
    image_perturbated = torch.Tensor(image_perturbated).to(device)
    # Imag_perturbated=toPIL(image_perturbated.squeeze(0))
    # Imag_perturbated.show()
    
    net.eval()
    
    print(net(image_perturbated))
    print(net(image.unsqueeze(0).to(device)))
    _,predictedLabel = torch.max(net(image_perturbated),1)
    print(predictedLabel)