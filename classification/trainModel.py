import torch
import torchvision
import torchvision.transforms as transforms
import MyModel
import testModel
import torch.nn as nn

# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.0005

# load MNIST dataset
train_data = torchvision.datasets.KMNIST(root='../ImageData', train=True, transform=transforms.ToTensor(),
                                         download=True)  # download training data set
test_data = torchvision.datasets.KMNIST(root='../ImageData', train=False, transform=transforms.ToTensor(),
                                        download=True)  # download testing data set

# define dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batch_size,
                                          shuffle=False)

# load model
net = MyModel.LeNet_my(num_classes).to(device)

# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = net(images)
        loss = criterion(outputs, labels)
        if loss.item() < 0.003:
            break

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Save the model checkpoint
torch.save(net.state_dict(), '../pretrainModel/ResNet_KMNIST.ckpt')
testModel.test(net, test_loader)
