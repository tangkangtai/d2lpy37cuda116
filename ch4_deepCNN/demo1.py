import torch
from torch import nn
from torch.nn import functional as F

# nn.Sequential定义了⼀种特殊的Module，即在PyTorch中表⽰⼀个块的类，它维护了⼀个由Module组
# 成的有序列表。注意，两个全连接层都是Linear类的实例，Linear类本⾝就是Module的⼦类

# 到⽬前为⽌，我们⼀直在通过net(X)调⽤我们的模型来获得模型的输出。
# 这实际上是net.__call__(X)的简写。
# 这个前向传播函数⾮常简单：它将列表中的每个块连接在⼀起，将每个块的输出作为下⼀个块的输⼊。
net = nn.Sequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

X = torch.rand(2, 20)
print(net(X))

#我们从零开始编写⼀个块。它包含⼀个多层感知机，其具有256个隐藏单元的隐藏层
# 和⼀个10维输出层。注意，下⾯的MLP类继承了表⽰块的类

class MLP(nn.Module):
    # ⽤模型参数声明层。这⾥，我们声明两个全连接的层
    def __init__(self):
    # 调⽤MLP的⽗类Module的构造函数来执⾏必要的初始化。
    # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)
    # # 定义模型的前向传播，即如何根据输⼊X返回所需的模型输出
    def forward(self, X):
# 注意，这⾥我们使⽤ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))

net = MLP()
print(net(X))

# 可以更仔细地看看Sequential类是如何⼯作的，回想⼀下Sequential的设计是为了把其他模块
# 串起来。为了构建我们⾃⼰的简化的MySequential，

# 我们只需要定义两个关键函数：
# 1. ⼀种将块逐个追加到列表中的函数。
# 2. ⼀种前向传播函数，⽤于将输⼊按追加块的顺序传递给块组成的“链条”。
# MySequential类提供了与默认Sequential类相同的功能。

"""
__init__函数将每个模块逐个添加到有序字典_modules中。
你可能会好奇为什么每个Module都有⼀个_modules属性？
以及为什么我们使⽤它⽽不是⾃⼰定义⼀个Python列表？简⽽⾔之，_modules的主要
优点是：在模块的参数初始化过程中，系统知道在_modules字典中查找需要初始化参数的⼦块
"""
class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        for idx, module in enumerate(args):
            # 这⾥，module是Module⼦类的⼀个实例。我们把它保存在'Module'类的成员
            # 变量 _modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X

net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(net(X))

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super(FixedHiddenMLP, self).__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=True)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使⽤创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复⽤全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1: # 运⾏了⼀个while循环，在L1范数⼤于1的条件下，将输出向量除以2，直到它满⾜条件为⽌。
            X /= 2
        return X.sum()

net = FixedHiddenMLP()
print(net(X))

# 可以混合搭配各种组合块的⽅法
class NestMLP(nn.Module):
    def __init__(self):
        super(NestMLP, self).__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))
