import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# 调⽤框架中现有的API来读取数据。我们将features和labels作为API的参数传递，并通过数据
# 迭代器指定batch_size。此外，布尔值is_train表⽰是否希望数据迭代器对象在每个迭代周期内打乱数据
def load_array(data_arrays, batch_size, is_train=True):
    """构造⼀个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 使⽤iter构造Python迭代器，并使⽤next从迭代器中获取第⼀项。
print(next(iter(data_iter)))

from torch import nn
# 在PyTorch中，全连接层在Linear类中定义。值得注意的是，我们将两个参数传递到nn.Linear中。第⼀
# 个指定输⼊特征形状，即2，第⼆个指定输出特征形状，输出特征形状为单个标量，因此为1。
net = nn.Sequential(nn.Linear(2, 1))
# 我们通过net[0]选择⽹络中的第⼀个图层，然后使⽤weight.data和bias.data⽅法访问参数。
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss() # 计算均⽅误差使⽤的是MSELoss类，也称为平⽅L2范数, 默认情况下，它返回所有样本损失的平均值。
# 当我们实例化⼀个SGD实例时，我们要指定优化的参数（可通过net.parameters()从我们的模型中获
# 得）以及优化算法所需的超参数字典。⼩批量随机梯度下降只需要设置lr值，这⾥设置为0.03。
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3

for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad() # 优化器梯度清零
        l.backward()  # 损失函数 反向传播 得到每个参数的梯度值（loss.backward()）
        trainer.step()  # 优化器 梯度下降执行一步参数更新（optimizer.step()）  更新优化器的学习率的
    l = loss(net(features), labels)
    print(f'epoch{epoch+1}, loss{l:f}')


w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)

