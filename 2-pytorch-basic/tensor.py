import torch

print(torch.__version__)
# print(torch.rand(2,3))
device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# other common initialization methods
print(torch.empty(size=(3, 3)))
print(torch.zeros((3, 3)))
print(torch.rand((3, 3)))
print(torch.eye(5, 5))
print(torch.arange(1, 10, 2))
# 1-10分100份
print(torch.linspace(1, 10, 100))
# 从正态分布
print(torch.empty(size=(2, 5)).normal_(mean=0, std=1))
# 从均匀分布
print(torch.empty(size=(1, 5)).uniform_(0, 1))
# 取对角线的元素
print(torch.diag(torch.ones(3)))

# how to initialize and convert to other types(int,float,double)

tensor = torch.arange(4)
print(tensor.shape)
print(tensor)
print(tensor.bool())  # boolean true or false
print(tensor.short())  # int 16
print(tensor.long())  # int64
print(tensor.half())  # float16
print(tensor.float())  # float32
print(tensor.double())  # float64

# array to tensor conversion and vice-versa

import numpy as np

np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()


x = torch.arange(9)
x_3x3 = x.view(3, 3)
x_3x3 = x.reshape(3, 3)
print(x_3x3)
y = x_3x3.t()
print(y.shape)
print(y.contiguous().view(9))

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(torch.cat((x1, x2), dim=0).shape)
print(torch.cat((x1, x2), dim=1).shape)

print(x2.view(-1).shape)

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)

z = x.permute(0, 2, 1)

x = torch.arange(10)
print(x.unsqueeze(0))
print(x.unsqueeze(1))
