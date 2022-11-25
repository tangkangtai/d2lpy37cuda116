import random
import torch
from d2l import torch as d2l
##⽣成数据集
# y = Xw + b + ϵ.     ，即ϵ服从均值为0的正态分布
def synthetic_data(w, b, num_examples):
    """⽣成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
#
# # features中的每⼀⾏都包含⼀个⼆维数据样本，labels中的每⼀⾏都包含⼀维标签值（⼀个标量）。
# print('features:', features[0], '\nlabel:', labels[0])
#
# #通过⽣成第⼆个特征features[:, 1]和labels的散点图，可以直观观察到两者之间的线性关系。
# d2l.set_figsize()
# d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
# d2l.plt.show()


# 定义⼀个data_iter函数，该函数接收批量⼤⼩、特征矩阵和标签向量作为输⼊，⽣
# 成⼤⼩为batch_size的⼩批量。每个⼩批量包含⼀组特征和标签。
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
# 读取第⼀个⼩批量数据样本并打印。每个批量的特征维度显⽰批量⼤⼩和输
# ⼊特征数。同样的，批量的标签形状与batch_size相等。

"""当我们运⾏迭代时，我们会连续地获得不同的⼩批量，直⾄遍历完整个数据集"""
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break


# 开始⽤⼩批量随机梯度下降优化我们的模型参数之前，我们需要先有⼀些参数。在下⾯的代码中，我
# 们通过从均值为0、标准差为0.01的正态分布中采样随机数来初始化权重，并将偏置初始化为0。
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 在初始化参数之后，我们的任务是更新这些参数，直到这些参数⾜够拟合我们的数据。每次更新都需要计算
# 损失函数关于模型参数的梯度。有了这个梯度，我们就可以向减⼩损失的⽅向更新每个参数。因为⼿动计算
# 梯度很枯燥⽽且容易出错，所以没有⼈会⼿动计算梯度

def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

"""在每⼀步中，使⽤从数据集中随机抽取的⼀个⼩批量，然后根据参数计算损失的梯度。接下来，朝着减少损
失的⽅向更新我们的参数。
下⾯的函数实现⼩批量随机梯度下降更新。
该函数接受模型参数集合、学习速率和批量⼤⼩作为输⼊。
每⼀步更新的⼤⼩由学习速率lr决定。因为我们计算的损失是⼀个批量样本的总和，
所以我们⽤批量⼤⼩（batch_size）来规范化步⻓，这样步⻓⼤⼩就不会取决于我们对批量⼤⼩的选择"""

def sgd(params, lr, batch_size):
    """⼩批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 在每个迭代周期（epoch）中，我们使⽤data_iter函数遍历整个数据集，并将训练数据集中所有样本都使
# ⽤⼀次（假设样本数能够被批量⼤⼩整除）。这⾥的迭代周期个数num_epochs和学习率lr都是超参数，分别设为3和0.03
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y) # X和y的⼩批量损失
        # 因为l形状是(batch_size,1)，⽽不是⼀个标量。l中的所有元素被加到⼀起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size) # 使⽤参数的梯度更新参数

    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch+1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')