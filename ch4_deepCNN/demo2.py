import torch
from torch import nn
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net(X))

# 参数访问
# 当通过Sequential类定义模型时，我们可以通过索引来访问模型的任意层。
# 这就像模型是⼀个列表⼀样，每层的参数都在其属性中
"""⾸先，这个全连接层包含两个参数，分别是该层的权重和偏置"""
print(net[2].state_dict()) # 以检查第⼆个全连接层的参数
# 每个参数都表⽰为参数类的⼀个实例。要对参数执⾏任何操作，⾸先我们需要访问底层的数值
"""参数是复合的对象，包含值、梯度和额外信息"""
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
"""还可以访问每个参数的梯度。在上⾯这个⽹络中，由于我们还没有调⽤反向传播，所以参数的梯度处于初始状态。"""
print(net[2].weight.grad == None) # True

# ⼀次性访问所有参数
"""我们将通过演⽰来⽐较访问第⼀个全连接层的参数和访问所有层"""
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

print(net.state_dict()['2.bias'].data)

def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net
rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet(X))
print(rgnet)

#  参数初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weigth, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
print(net[0].weight.data[0], net[0].bias.data[0])

# 还可以将所有参数初始化为给定的常数，⽐如初始化为1。
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0])

# Xavier初始化⽅法初始化第⼀个神经⽹络层，然后将第三个神经⽹络层初始化为常量值42。

def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

# 参数绑定
# 希望在多个层间共享参数：我们可以定义⼀个稠密层，然后使⽤它的参数来设置另⼀个层的参数

shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8),
                    nn.ReLU(),
                    shared,
                    nn.ReLU(),
                    shared,
                    nn.ReLU(),
                    nn.Linear(8, 1))
print(net(X))
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同⼀个对象，⽽不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
# 表明第三个和第五个神经⽹络层的参数是绑定的。它们不仅值相等，⽽且由相同的张量表⽰
# 当参数绑定时，梯度会发⽣什么情况？
# 答案是由于模型参数包含梯度，因此在反向传播期间第⼆个隐藏层（即第三个神经⽹络层）和第三个隐藏层（即第五个神经⽹络层）的梯度会加在⼀起。