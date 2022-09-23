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

# input_shape = [时间步数, 批量大小, 特征维度] = [num_steps(seq_length), batch_size, input_dim]
# 在前向计算后会分别返回输出和隐藏状态h，其中输出指的是隐藏层在各个时间步上计算并输出的隐藏状态，它们通常作为后续输出层的输⼊。
# 需要强调的是，该“输出”本身并不涉及输出层计算，形状为(时间步数, 批量大小, 隐藏单元个数)；这个隐藏单元个数就是hidden_size。
# 隐藏状态指的是隐藏层在最后时间步的隐藏状态：当隐藏层有多层时，每⼀层的隐藏状态都会记录在该变量中；也就是说"输出"是有seq_length个，但是隐藏层，有几层就几个
# 对于像⻓短期记忆（LSTM），隐藏状态是⼀个元组(h, c)，即hidden state和cell state(此处普通rnn只有一个值)，隐藏状态h的形状为(层数, 批量大小,隐藏单元个数)


import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

# rnn可以用于序列生成序列，假设现在我们有个"helliosss"需要变换成"qostis"
# 首先根据单词构建一个此表，用这个将输入变成id，再转化为tensor放入到模型中，这个例子中是先变成了onehot，再将数据变成float格式，然后才放
# 到model中，其实可以直接用nn.embedding，根据索引直接取出词向量，但是要使用detach()，否则这个embedding会参与计算，而期间又会被销毁，因此就会报错。

idx2char = ['e', 'h', 'l', 'o', 's', 't', 'i', "q"]
input = 'helliosss'
x_idx = torch.tensor([idx2char.index(i) for i in input])
# x_data = F.one_hot(x_idx, 4).reshape(-1, 1, 4).float()  # RNNCell
input_dim = len(idx2char)
x_data = F.one_hot(x_idx, input_dim).float().reshape(len(input), 1, input_dim)  # seq_len, batch, input_dim

# 下面是用embedding的做法
# e = nn.Embedding(50, 8)
# x_data = e(x_idx).view(len(x_idx), 1, len(idx2char)).detach()
# 如果不用detach分离，会发现x_data并不是叶子节点
# print(x_data.is_leaf)

# y_data = torch.tensor([idx2char.index(i) for i in 'ohlol']).reshape(-1, 1) ## RNNCell
target = "qostis"
y_data = torch.tensor([idx2char.index(i) for i in target])


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
        return y.reshape(-1, self.hidden_size), test[-1]
        # return self.rnncell(x, hidden)  # RNNCell

    # def init_hidden(self):  # RNNCell
    #     return torch.zeros(self.batch_size, self.hidden_size)


# 这里，输出的维度要大于字典的大小，因为输出的维度的一堆值就是代表这字典每个字的概率，如果不大于等于字典的大小，那么就无法进行loss的计算，因为会出现索引的越界
net = Module(input_dim, len(idx2char))
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
    y, test = net(x_data)
    # y的个数取决于输入字符串的长度，如果大于target字符串，则只取target字符串的长度，这样才能进行loss计算。
    # 问题来了，如果要想短的字符串翻译成长字符串，抱歉，现在这里的辣鸡代码还不能实现，要看seq2seq。这里只是个简单的使用RNN的代码罢了。
    loss = criterion(y[:len(y_data)], y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _, idx = torch.max(y, dim=1)
    pre = ''.join(idx2char[i] for i in idx)
    print(test == y)  # 会发现，最后一层和y是相同的，也就是最后的隐藏层输出，是和总的输出的最后一个元素是完全一样的。
    print('{}, 第{}轮, loss为{:.4f}'.format(pre[:len(y_data)], epoch + 1, loss.item()))
    # 如果发现已经一样了，就直接停掉程序，没有任何其他的用处
    if pre[:len(y_data)] == target:
        break
