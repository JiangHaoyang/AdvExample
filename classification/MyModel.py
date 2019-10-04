import torch.nn as nn

class Model_1(nn.Module):
    def __init__(self):
        super(Model_1,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 20, kernel_size = 5, stride = 1, ),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d( in_channels = 20, out_channels = 50, kernel_size = 5, stride = 1, ),
            nn.BatchNorm2d( 50 ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear( 50*4*4, 500 ),
            nn.BatchNorm1d(500),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(500, 10)
        )
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0),-1)
        out = self.layer3(out)
        out = self.layer4(out)
        return out