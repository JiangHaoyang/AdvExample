import torch
import torchvision
import torchvision.transforms as transforms
import sys
sys.path.append('..')
import Models

def experimentSetup(modelName='LeNet',dataSet='MNIST',pretrain=False):
    # device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ####################################################################
    # select classification network
    net = None
    if modelName == 'LeNet': net = Models.MyModel.Model_1(num_classes=10) #MNIST
    if modelName == 'VGG16': Models.vgg.VGG('VGG16',num_classes=10) #CIFAR10
    if modelName == 'ResNet18': net = Models.resnet.ResNet18(num_classes=10) #CIFAR10
    pretrainModel = {'LeNet':'LeNet_MNIST_ckpt.pth',
                     'VGG16':'Vgg16_Cifar10_ckpt.pth',
                     'ResNet18':'ResNet18_Cifar10_ckpt.pth'}
    if pretrain:
        checkpoint = torch.load('../pretrainModel/'+pretrainModel[modelName])
        net.load_state_dict(checkpoint['net'])
    net = net.to(device)
    ####################################################################
    # select dataset
    if dataSet == 'MNIST':
        transform_MNIST = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.MNIST(root = '../ImageData',
                                            train = True,
                                            download = True,
                                            transform = transform_MNIST)
        testset = torchvision.datasets.MNIST(root = '../ImageData',
                                        train = False,
                                        download = True,
                                        transform = transform_MNIST)
    if dataSet == 'CIFAR10':
        transform_train_CIFAR = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test_CIFAR = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.CIFAR10(root = '../ImageData',
                                                train = True,
                                                download = True,
                                                transform = transform_train_CIFAR)
        testset = torchvision.datasets.CIFAR10(root = '../ImageData',
                                            train = False,
                                            download = True,
                                            transform = transform_test_CIFAR)
        # CIFAR10
        # classes = ('plane', 'car', 'bird', 'cat', 'deer',
        #            'dog', 'frog', 'horse', 'ship', 'truck')
    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size = 128,
                                            shuffle = True)
    testloader = torch.utils.data.DataLoader(testset,
                                            batch_size = 128,
                                            shuffle = False)

    return [net, testset, trainset, testloader, trainloader, device]
