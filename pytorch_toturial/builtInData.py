import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

to_pil_image = transforms.ToPILImage() #transform function: from tensor data to PIL data

#define how an PIL image transforms into the network input
inputTransformer=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])


mnist_data_train = torchvision.datasets.MNIST(root='./ImageData', train = True, transform = transforms.ToTensor(), download=True) #download training data set
mnist_data_test = torchvision.datasets.MNIST(root='./ImageData', train = False, transform = transforms.ToTensor(), download=True) #download testing data set
#all transforms functions are applied here


[image,label]=mnist_data_train[0] #extract a pair of data
print(image.shape)
#image is in float form now

# img=to_pil_image(image) #conver the tensor data to PIL data
# img.show()

# img_np=np.array(img) #conver the PIL data into numpy interget form
# print(img_np.shape)
# print(label)
