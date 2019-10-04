import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import os
import sys
sys.path.append('..')
import Models

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')

transform_train_CIFAR = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test_CIFAR = transforms.Compose([transforms.ToTensor()])
transform_MNIST = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root = '../ImageData',
                                        train = True,
                                        download = True,
                                        transform = transform_train_CIFAR)
testset = torchvision.datasets.CIFAR10(root = '../ImageData',
                                       train = False,
                                       download = True,
                                       transform = transform_test_CIFAR)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size = 128,
                                          shuffle = True)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size = 128,
                                         shuffle = False)
# CIFAR10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = Models.vgg.VGG('VGG16') #CIFAR10
# net = Models.resnet.ResNet18() #CIFAR10
# net = Models.MyModel.Model_1() #MNIST

net = net.to(device)

# Hyper parameters
start_epoch = 0
num_epochs = 50
learning_rate = 0.05
best_acc = 0
resumeTrain = False
if resumeTrain:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    pass

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    # set to train model
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _,predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        # print progress_bar
        print('batch',batch_idx,'/',len(trainloader),
              'Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
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
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

for epoch in range(start_epoch, start_epoch+num_epochs):
    train(epoch)
    test(epoch)



















