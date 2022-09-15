'''
    这是个最简单的例子，输入都是固定的大小
'''
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from models.model_utils import *
dtype = torch.FloatTensor

sentences = ["i like dog", "i love coffee", "i hate milk"]

word_list = " ".join(sentences).split()  # ['i', 'like', 'dog', 'dog', 'i', 'love', 'coffee', 'i', 'hate', 'milk']
word_list = list(set(word_list))  # ['i', 'like', 'dog', 'love', 'coffee', 'hate', 'milk']
word_dict = {w: i for i, w in
             enumerate(word_list)}  # {'i':0, 'like':1, 'dog':2, 'love':3, 'coffee':4, 'hate':5, 'milk':6}
number_dict = {i: w for i, w in
               enumerate(word_list)}  # {0:'i', 1:'like', 2:'dog', 3:'love', 4:'coffee', 5:'hate', 6:'milk'}
n_class = len(word_dict)  # number of Vocabulary, just like |V|, in this task n_class=7

# NNLM(Neural Network Language Model) Parameter
n_step = len(
    sentences[0].split()) - 1  # n-1 in paper, look back n_step words and predict next word. In this task n_step=2
n_hidden = 5  # h in paper，隐藏层神经元个数
m = 20  # m in paper, word embedding dim


def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]  # [0, 1], [0, 3], [0, 5]
        target = word_dict[word[-1]]  # 2, 4, 6

        input_batch.append(input)  # [[0, 1], [0, 3], [0, 5]]
        target_batch.append(target)  # [2, 4, 6]

    return input_batch, target_batch


input_batch, target_batch = make_batch(sentences)
input_batch = torch.LongTensor(input_batch)
target_batch = torch.LongTensor(target_batch)

dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset=dataset, batch_size=16, shuffle=True)


class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        # m 词向量的维度，一般大于 50
        self.C = nn.Embedding(n_class, m)
        # H 隐藏层的 weight
        self.H = nn.Parameter(torch.randn(n_step * m, n_hidden).type(dtype))
        # W 输入层到输出层的 weight
        self.W = nn.Parameter(torch.randn(n_step * m, n_class).type(dtype))
        # d 隐藏层的 bias
        self.d = nn.Parameter(torch.randn(n_hidden).type(dtype))
        # U 输出层的 weight
        self.U = nn.Parameter(torch.randn(n_hidden, n_class).type(dtype))
        # b 输出层的 bias
        self.b = nn.Parameter(torch.randn(n_class).type(dtype))

    def forward(self, X):
        '''
        X: [batch_size, n_step]
        '''
        X = self.C(X)  # [batch_size, n_step] => [batch_size, n_step, m]
        X = X.view(-1, n_step * m)  # [batch_size, n_step * m]
        hidden_out = torch.tanh(self.d + torch.mm(X, self.H))  # [batch_size, n_hidden]
        output = self.b + torch.mm(X, self.W) + torch.mm(hidden_out, self.U)  # [batch_size, n_class]
        return output


model = NNLM()
total_paramters(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(5000):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        output = model(batch_x)

        # output : [batch_size, n_class], batch_y : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, batch_y)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

# Predict
predict = model(input_batch).data.max(1, keepdim=True)[1]

# Test
print([sen.split()[:n_step] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])
