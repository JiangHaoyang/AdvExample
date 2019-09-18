import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# Build a computational graph.
y = w * x + b    # y = 2 * x + 3

# Compute gradients.
y.backward()

# Print out the gradients.
print(x.grad)    # x.grad = 2 
print(w.grad)    # w.grad = 1 
print(b.grad)    # b.grad = 1

x = torch.randn(10,3)
y = torch.randn(10,2)

linear=nn.Linear(3,2)
print('w:',linear.weight)
print('b:',linear.bias)

criterion = nn.MSELoss()
optimizer = torch.optim