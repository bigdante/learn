# 输入的参数列表包括:
#
# input_size 输入数据的特征维数，通常就是embedding_dim(词向量的维度)
# hidden_size　LSTM中隐层的维度
# num_layers　循环神经网络的层数
# bias　用不用偏置，default=True
# batch_first 这个要注意，通常我们输入的数据shape=(batch_size,seq_length,embedding_dim),而batch_first默认是False,
# 所以我们的输入数据最好送进LSTM之前将batch_size与seq_length这两个维度调换
# dropout　默认是0，代表不用dropout
# bidirectional默认是false，代表不用双向LSTM
# 输入数据包括input,(h_0,c_0):
#
# input就是shape=(seq_length,batch_size,input_size)的张量
# h_0是shape=(num_layers*num_directions,batch_size,hidden_size)的张量，它包含了在当前这个batch_size中每个句子的初始隐藏状态。
# 其中num_layers就是LSTM的层数。如果bidirectional=True,num_directions=2,否则就是１，表示只有一个方向。
# c_0和h_0的形状相同，它包含的是在当前这个batch_size中的每个句子的初始细胞状态。h_0,c_0如果不提供，那么默认是０。
# 输出数据包括output,(h_n,c_n):
#
# output的shape=(seq_length,batch_size,num_directions*hidden_size),
# 它包含的是LSTM的最后一时间步的输出特征(h_t),ｔ是batch_size中每个句子的长度。
# h_n.shape==(num_directions * num_layers,batch,hidden_size)
# c_n.shape==h_n.shape
# h_n包含的是句子的最后一个单词（也就是最后一个时间步）的隐藏状态，c_n包含的是句子的最后一个单词的细胞状态，所以它们都与句子的长度seq_length无关。
# output[-1]与h_n是相等的，因为output[-1]包含的正是batch_size个句子中每一个句子的最后一个单词的隐藏状态，注意LSTM中的隐藏状态其实就是输出，cell state细胞状态才是LSTM中一直隐藏的，记录着信息
#

# coding: utf-8
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

input_dim = 28  # 输入维度
hidden_dim = 100  # 隐层的维度
layer_dim = 1  # 1层LSTM
output_dim = 10  # 输出维度
BATCH_SIZE = 32  # 每批读取的
EPOCHS = 10  # 训练10轮


trainsets = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())

testsets = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=trainsets, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testsets, batch_size=BATCH_SIZE, shuffle=True)


class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # layer_dim, batch_size, hidden_dim
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # 初始化cell, state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # 分离隐藏状态，避免梯度爆炸
        lstm_out, (h_n, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        print("-" * 10)
        print("lstm_out.shape", lstm_out.shape)
        print("lstm_out[:, -1].shape", lstm_out[:, -1].shape)
        print("-" * 10)
        print("h_n.shape", h_n.shape)
        print("-" * 10)
        feature_map = torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=-1)
        print("feature_map.shape", feature_map.shape)
        print("-" * 10)
        print("lstm_out[:, -1]", lstm_out[:, -1])
        print("-" * 10)
        print("feature_map", feature_map)
        print("-" * 10)
        out = self.fc(feature_map)
        print("out", out.shape)
        return out

model = LSTM_Model(input_dim, hidden_dim, layer_dim, output_dim)

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 模型训练
sequence_dim = 28  # 序列长度
loss_list = []
accuracy_list = []
iteration_list = []  # 迭代次数

iter = 0
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        model.train()
        images = images.view(-1, sequence_dim, input_dim).requires_grad_().to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # 前向传播
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 计数器加1
        iter += 1
        # 模型验证
        if iter % 500 == 0:
            model.eval()  # 声明
            # 计算验证的accuracy
            correct = 0.0
            total = 0.0
            # 迭代测试集，获取数据、预测
            for images, labels in test_loader:
                images = images.view(-1, sequence_dim, input_dim).to(device)
                # 模型预测
                outputs = model(images)
                # 获取预测概率最大值的下标
                predict = torch.max(outputs.data, 1)[1]
                # 统计测试集的大小
                total += labels.size(0)
                # 统计判断预测正确的数量
                if torch.cuda.is_available():
                    correct += (predict.gpu() == labels.gpu()).sum()
                else:
                    correct += (predict == labels).sum()

                    # 计算accuracy
            accuracy = correct / total * 100
            loss_list.append(loss.data)
            accuracy_list.append(accuracy)
            iteration_list.append(iter)
            # 打印信息
            print("loos:{}, Loss:{}, Accuracy:{}".format(iter, loss.item(), accuracy))

plt.plot(iteration_list, loss_list)
plt.xlabel("Number of Iteration")
plt.ylabel("Loss")
plt.title("LSTM")
plt.show()


plt.plot(iteration_list, accuracy_list, color='r')
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("LSTM")
plt.show()
