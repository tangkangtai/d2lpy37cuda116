
import torch
from torch import nn
from d2l import torch as d2l
# 卷积层是个错误的叫法，因为它所表达的运算其实是互相关运算（cross-correlation）
def corr2d(X, K):
    """计算⼆维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()

    return Y

# X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
# K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
# Y = corr2d(X, K)
# print(Y)

# 卷积层对输⼊和卷积核权重进⾏互相关运算，并在添加标量偏置之后产⽣输出
# class Conv2D(nn.Module):
#     def __init__(self, kernel_size):
#         super(Conv2D, self).__init__()
#         self.weight = nn.Parameter(torch.rand(kernel_size))
#         self.bias = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         return corr2d(x, self.weight) + self.bias

# 我们先构造⼀个卷积层，并将其卷积核初始化为随机张量。
# 接下来，在每次迭代中，我们⽐较Y与卷积层输出的平⽅误差，然后计算梯度
# 来更新卷积核。
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

# 这个⼆维卷积层使⽤四维输⼊和输出格式（批量⼤⼩、通道、⾼度、宽度），
# 其中批量⼤⼩和通道数都为1

X = torch.ones((6, 8))
X[:, 2:6] = 0
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)

print(X.shape)
print(Y.shape)

X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
#
lr = 3e-2
for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if(i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')

print(conv2d.weight.data.reshape((1, 2)))



