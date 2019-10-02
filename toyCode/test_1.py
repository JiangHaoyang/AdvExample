import numpy as np
import torch
import PIL
from PIL import Image
import torchvision.transforms as transforms

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
imag = Image.fromarray(a.astype('uint8'))

toPIL = transforms.ToPILImage()
toTensor = transforms.ToTensor()

print(np.array(imag).max())
print(toTensor(imag).max())
print(9/255)