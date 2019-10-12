import torch

x = torch.autograd.Variable(torch.Tensor([5]),requires_grad = True)
y = x**2

grad_x = torch.autograd.grad(y, x,create_graph=True)
print(grad_x) # dy/dx = 2 * x

grad_grad_x = torch.autograd.grad(grad_x[0],x)
print(grad_grad_x)