

import torch
from torch import nn
from torch.nn import functional as F
# 读写⽂件
# 希望保存训练的模型，以备将来在各种环境中使⽤（⽐如在部署中进⾏预测）。
# 此外，当运⾏⼀个耗时较⻓的训练过程时，最佳的做法是定期保存中间结果，以确保在服务器电源被不⼩⼼断掉时

############################# 加载和保存张量
x = torch.arange(4)
"""对于单个张量，我们可以直接调⽤load和save函数分别读写它们。
这两个函数都要求我们提供⼀个名称，save要求将要保存的变量作为输⼊"""
torch.save(x, 'x-file')

x2 = torch.load('x-file')
print(x2)

# 可以存储⼀个张量列表，然后把它们读回内存。
y = torch.zeros(4)
torch.save([x, y], 'x-file')
x2, y2 = torch.load('x-file')
print(x2,y2)

# 可以写⼊或读取从字符串映射到张量的字典。当我们要读取或写⼊模型中的所有权重时，这很⽅便。
mydict = {'x':x, 'y':y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(20 ,256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
# 将模型的参数存储在⼀个叫做“mlp.params”的⽂件中
torch.save(net.state_dict(), 'mlp.params')

# 为了恢复模型，我们实例化了原始多层感知机模型的⼀个备份。这⾥我们不需要随机初始化模型参数，⽽是
# 直接读取⽂件中存储的参数
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()

Y_clone = clone(X)
print(Y_clone == Y)
