import torch
from torch import nn

e = nn.Embedding(50,10)
a = torch.LongTensor([[1,2]])
print(a.shape)
print(e(a))

