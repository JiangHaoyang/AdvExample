import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import os
import sys
sys.path.append('..')
import Models
import foolbox
import attackModel

# device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

############################################################################################################## select model
# select classification network
print('==> Building model..')
# net = Models.vgg.VGG('VGG16',num_classes=10) #CIFAR10
# net = Models.resnet.ResNet18(num_classes=10) #CIFAR10
net = Models.MyModel.Model_1(num_classes=10) #MNIST
net = net.to(device)
############################################################################################################## select dataset
transform_train_CIFAR = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test_CIFAR = transforms.Compose([transforms.ToTensor()])
transform_MNIST = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root = '../ImageData',
                                        train = True,
                                        download = True,
                                        transform = transform_MNIST)
testset = torchvision.datasets.MNIST(root = '../ImageData',
                                       train = False,
                                       download = True,
                                       transform = transform_MNIST)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size = 1,
                                          shuffle = True)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size = 128,
                                         shuffle = False)
# CIFAR10
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Hyper parameters
start_epoch = 0
num_epochs = 200
learning_rate = 0.001
best_acc = 0
eps = 0.1
resumeTrain = False
if resumeTrain:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/LeNet_MNIST_adv.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    pass

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

net.eval()
fmodel = foolbox.models.PyTorchModel(net, bounds=(0,1),num_classes=10)
attack = foolbox.attacks.ProjectedGradientDescent(fmodel)


def adversarialTrain(epoch):
    print('\nEpoch: %d' % epoch)
    # set to train model
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        ################################################ calculate the perturbation
        x = inputs.clone()
        x = torch.autograd.Variable(x,requires_grad=True)
        outputs_natural = net(x)
        pertLoss = criterion(outputs_natural,labels)
        pertLoss.backward(retain_graph=True)
        pert = x.grad

        optimizer.zero_grad()
        outputs_adversary = net(inputs + eps*pert)
        loss = criterion(outputs_natural, labels)+1000*torch.abs(1*criterion(outputs_natural, labels) - 1*criterion(outputs_adversary, labels))
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _,predicted_natural = outputs_natural.max(1)
        _,predicted_adversay = outputs_adversary.max(1)
        total += (labels.size(0)*2)
        correct += predicted_natural.eq(labels).sum().item()
        correct += predicted_adversay.eq(labels).sum().item()
        # print progress_bar
        print('batch',batch_idx,'/',len(trainloader),
              'Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def adversarialTrain_secondDerivative(epoch):
    print('\nEpoch: %d' % epoch)
    # set to train model
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        ################################################ calculate the perturbation
        x = inputs.clone()
        x = torch.autograd.Variable(x,requires_grad=True)
        outputs_natural = net(x)
        pertLoss = criterion(outputs_natural,labels)
        pert = torch.autograd.grad(pertLoss,x,create_graph=True)
        pert = pert[0]
        pertNorm = pert.norm()

        optimizer.zero_grad()
        outputs_adversary = net(inputs + eps*pert)
        loss = criterion(outputs_natural,labels) + 100 * pertNorm
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _,predicted_natural = outputs_natural.max(1)
        _,predicted_adversay = outputs_adversary.max(1)
        total += (labels.size(0)*2)
        correct += predicted_natural.eq(labels).sum().item()
        correct += predicted_adversay.eq(labels).sum().item()
        # print progress_bar
        print('batch',batch_idx,'/',len(trainloader),
              'Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def adversarialTrain_PGD(epoch):
    print('\nEpoch: %d' % epoch)
    # set to train model
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.squeeze(0)
        net.eval()
        adversarial = attack(inputs.numpy(),int(labels.numpy()))
        adversarial = torch.from_numpy(adversarial).to(device)
        net.train()
        print('this function is not finished')
        input()
        optimizer.zero_grad()
        outputs_natural = net(adversarial)
        loss = criterion(outputs_natural,labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _,predicted = outputs_natural.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        # print progress_bar
        print('batch',batch_idx,'/',len(trainloader),
              'Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        pass
    pass

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _,predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            # print progress_bar
            print('batch',batch_idx,'/',len(testloader),
                  'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
    # Save checkpoint
    acc = 100.*correct/total
    if acc >= best_acc:
        print('Saving...')
        state = {
                'net': net.state_dict(),
                'acc':acc,
                'epoch':epoch}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_adv.pth')
        best_acc = acc

if __name__ == "__main__":
    for epoch in range(start_epoch, start_epoch+num_epochs):
        adversarialTrain_PGD(epoch)
        test(epoch)