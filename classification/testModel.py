import torch
import torchvision
import torchvision.transforms as transforms
import MyModel
import torch.nn as nn

def test(net,dataLoader):
    # device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images,labels in dataLoader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _,predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


if __name__ == '__main__':
    # device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load model
    net = MyModel.LeNet_my(10).to(device)
    # load pretrained parameters
    checkpoint = torch.load('../pretrainModel/ResNet_KMNIST.ckpt',map_location = device)
    net.load_state_dict(checkpoint)

    # load data
    test_data = torchvision.datasets.KMNIST(root='../ImageData', train=False, transform=transforms.ToTensor(),
                                            download=True)  # download testing data set
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                batch_size=10,
                                                shuffle=False)
    
    test(net,test_loader)