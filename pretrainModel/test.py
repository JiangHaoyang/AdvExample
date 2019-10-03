import torch
import resnet
import torch.backends.cudnn as cudnn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = resnet.ResNet18().to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint = torch.load('ResNet18_ckpt.pth')
net.load_state_dict(checkpoint['net'])

test = torch.rand([1,3,32,32])
net.eval()
out = net(test)
print(out)