# softmax回归的简洁实现
import torch
from torch import nn
from d2l import torch as d2l
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


# 初始化模型参数
# # PyTorch不会隐式地调整输⼊的形状。因此，
# # 我们在线性层前定义了展平层（flatten），来调整⽹络输⼊的形状
# 我们只需在Sequential中添加⼀个带有10个输出的全连接层
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')

#使⽤学习率为0.1的⼩批量随机梯度下降作为优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()