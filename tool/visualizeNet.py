import sys
sys.path.append("..")
import classification.MyModel as MyModel
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

class visualizeImage():
    def __init__(self,centerData,dir1,dir2,net):
        self.center = centerData
        self.d1 = dir1
        self.d2 = dir2
        allImage = torch.zeros(torch.Size([10000])+self.center.size()).to(centerData.device)
        for x in range(100):
            for y in range(100):
                allImage[x*100+y,:,:]= self.dataPointGenerator(x,y)
        self.validmap = self.validMap(allImage)
        net.eval()
        self.netOut = net(allImage)

    def probe(self,x,y):
        image = self.dataPointGenerator(x,y).unsqueeze(0)
        return net(image)
        pass

    def dataPointGenerator(self,x,y):
        return self.center + 0.25*(x-50)*self.d1 + 0.25*(y-50)*self.d2
        pass

    
    def hotMap(self):
        map = torch.zeros([100,100]).to(self.netOut.device)
        for x in range(100):
            for y in range(100):
                map[x,y] = self.netOut[x*100+y,:].max()
        return map
        pass
    
    def boundMap(self):
        map = torch.zeros([100,100]).to(self.netOut.device)
        for x in range(100):
            for y in range(100):
                map[x,y] = self.netOut[x*100+y,:].argmax()
        return map
        pass

    def validMap(self,images):
        map = (images.max(2)[0].max(2)[0] < 1.01) * (images.min(2)[0].min(2)[0] > -0.1)
        print(map.max())
        map = map.reshape([100,100])
        return map
        pass

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
    
    # set the center of the map
    centerImage,label = test_data[84]
    centerImage = centerImage.to(device)
    print(label)

    randomDir = False
    if randomDir == True:
        # set two random direction vector, dir1 and dir2 are Orthogonal
        dir1 = torch.randn(centerImage.size()).to(device)
        dir1 = dir1 / dir1.norm()
        dir2 = torch.randn(centerImage.size()).to(device)
        tmp = (dir1 * dir2).sum() * dir1
        dir2 = dir2 - tmp
        dir2 = dir2 / dir2.norm()
    else:
        dir1 = torch.rand(centerImage.size()).to(device)
        dir2 = torch.rand(centerImage.size()).to(device)
        for i in range(100):
            image,label = test_data[i]
            if label == 1:
                dir1 = image.to(device)-centerImage
            if label == 7:
                dir2 = image.to(device)-centerImage
    print((dir1-dir2).norm())

    dir1 = dir1 / dir1.norm()
    dir2 = dir2 / dir2.norm()


    map1 = visualizeImage(centerImage,dir1,dir2,net)

    hotmap = map1.hotMap()
    hotmap[map1.validmap == False] = hotmap.min()
    hotmap -= hotmap.min()
    hotmap = hotmap/hotmap.max()

    hotImage = toPIL((hotmap).cpu())
    hotImage.save('hotImage.png',quality = 95)

    boundmap = map1.boundMap()
    boundImage = ((boundmap+1)*25)
    boundImage[map1.validmap == False] = 0
    boundImage = toPIL(boundImage.cpu())
    boundImage.save('boundImage.png',quality = 95)