# nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity=tanh, bias=True, batch_first=False, dropout=0, bidirectional=False)
#
# 参数说明
#
# input_size输入特征的维度， 一般rnn中输入的是词向量，那么 input_size 就等于一个词向量的维度
# hidden_size隐藏层神经元个数，或者也叫输出的维度（因为rnn输出为各个时间步上的隐藏状态）
# num_layers网络的层数
# nonlinearity激活函数
# bias是否使用偏置
# batch_first输入数据的形式，默认是 False，就是这样形式，(seq(num_step), batch, input_dim)，也就是将序列长度放在第一位，batch 放在第二位
# dropout是否应用dropout, 默认不使用，如若使用将其设置成一个0-1的数字即可
# birdirectional是否使用双向的 rnn，默认是 False

# 输入输出shape
#
# input_shape = [时间步数, 批量大小, 特征维度] = [num_steps(seq_length), batch_size, input_dim]
# 在前向计算后会分别返回输出和隐藏状态h，其中输出指的是隐藏层在各个时间步上计算并输出的隐藏状态，它们通常作为后续输出层的输⼊。需要强调的是，该“输出”本身并不涉及输出层计算，形状为(时间步数, 批量大小, 隐藏单元个数)；隐藏状态指的是隐藏层在最后时间步的隐藏状态：当隐藏层有多层时，每⼀层的隐藏状态都会记录在该变量中；对于像⻓短期记忆（LSTM），隐藏状态是⼀个元组(h, c)，即hidden state和cell state(此处普通rnn只有一个值)隐藏状态h的形状为(层数, 批量大小,隐藏单元个数)

# 输入输出shape
#
# input_shape = [时间步数, 批量大小, 特征维度] = [num_steps(seq_length), batch_size, input_dim]
# 在前向计算后会分别返回输出和隐藏状态h，其中输出指的是隐藏层在各个时间步上计算并输出的隐藏状态，它们通常作为后续输出层的输⼊。需要强调的是，该“输出”本身并不涉及输出层计算，形状为(时间步数, 批量大小, 隐藏单元个数)；隐藏状态指的是隐藏层在最后时间步的隐藏状态：当隐藏层有多层时，每⼀层的隐藏状态都会记录在该变量中；对于像⻓短期记忆（LSTM），隐藏状态是⼀个元组(h, c)，即hidden state和cell state(此处普通rnn只有一个值)隐藏状态h的形状为(层数, 批量大小,隐藏单元个数)


import torch
from torch.utils import data
from torch import nn
from torch import optim
import torch.nn.functional as F
# For plotting
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# RNN hello -> ohlol

idx2char = ['e', 'h', 'l', 'o', 's', 't', 'i',"q"]
x_idx = torch.tensor([idx2char.index(i) for i in 'hellio'])
# x_data = F.one_hot(x_idx, 4).reshape(-1, 1, 4).float()  # RNNCell
z = F.one_hot(x_idx, 8)
x_data = F.one_hot(x_idx, 8).float().reshape(6, 1, len(idx2char))  # 希望你能发现些什么, 诸如shape, type
print(x_data.shape)
# y_data = torch.tensor([idx2char.index(i) for i in 'ohlol']).reshape(-1, 1) ## RNNCell
y_data = torch.tensor([idx2char.index(i) for i in 'sehoo'])
print(y_data)


class Module(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size=1, num_layer=1):
        super(Module, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        # rnncell只有单层，所以h的参数就不一样
        # self.rnncell = nn.RNNCell(input_size, hidden_size)  # RNNCell
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=self.num_layer)

        self.fc = nn.Linear(5, 5)
    def forward(self, x):
        hidden = torch.zeros(self.num_layer, self.batch_size, self.hidden_size)
        y, test = self.rnn(x, hidden)
        d = test[-1]
        c = self.fc(test[-1])
        return y.reshape(-1, self.hidden_size), test[-1],c
        # return self.rnncell(x, hidden)  # RNNCell

    # def init_hidden(self):  # RNNCell
    #     return torch.zeros(self.batch_size, self.hidden_size)

print(x_data.shape[2])
net = Module(x_data.shape[2], len("sehoo"))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.1)
num_epochs = 100

for epoch in range(num_epochs):
    # RNNCell
    # loss = torch.tensor([0.])
    # hidden = net.init_hidden()  # RNNCell
    # for x, y in zip(x_data, y_data):
    #     # _y, hidden = net(x, hidden)
    #     loss += criterion(hidden, y)
    #     _, idx = torch.max(hidden, dim=1)
    #     print(idx2char[idx], end='')
    y, test,c = net(x_data)
    loss = criterion(c.squezze(), y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _, idx = torch.max(y, dim=1)
    pre = ''.join(idx2char[i] for i in idx)
    print(test.reshape(-1, len(idx2char)) == y)  # 希望你能发现些什么，最后一层和y是相同的
    print('{}, 第{}轮, loss为{:.4f}'.format(pre, epoch + 1, loss.item()))
