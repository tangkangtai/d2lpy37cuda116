import torch
from d2l import torch as d2l

# 当输⼊为负时，ReLU函数的导数为0，⽽当输⼊为正时，ReLU函数的导数为1
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
# y = torch.relu(x)
# d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))

# sigmoid通常称为挤压函数（squashing function）：它将范围（-inf, inf）中的任意输⼊压缩到区间（0, 1）中的某个值：
# sigmoid = 1 / 1 + exp(-x)


y = torch.sigmoid(x)
# d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))

# 当输⼊为0时，sigmoid函数的导数达到最⼤值0.25；⽽输⼊在任⼀⽅向上越远离0点时，导数越接近0。
# x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))


# tanh函数
# tanh(双曲正切)函数也能将其输⼊压缩转换到区间(-1, 1)上
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
x.grad.data.zero_()
# 当输⼊接近0时，tanh函数的导数接近最⼤值1。与我们在sigmoid函数图像
# 中看到的类似，输⼊在任⼀⽅向上越远离0点，导数越接近0。
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))

