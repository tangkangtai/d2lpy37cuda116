import torch
from torch import nn
from d2l import torch as d2l

# 注意⼒汇聚：Nadaraya-Watson 核回归
# 查询（⾃主提⽰）和键（⾮⾃主提⽰）之间的交互形成了注意⼒汇聚，注意⼒汇聚有选择地聚合了值（感官输⼊）以⽣成最终的输出
#
# ⽣成数据集
"""给定的成对的“输⼊－输出”数据集 {(x1, y1), . . . ,(xn, yn)}，如何学习f来预测任意新输⼊x的输出yˆ = f(x)？"""
# yi = 2 sin(xi) + x0^i.8 + ϵ 其中ϵ服从均值为0和标准差为0.5的正态分布。我们⽣成了50个训练样本和50个测试样本
n_train = 50
# # torch.sort排序后，返回排序后的tensor,和排序好了的元素对应之前的下标
x_train, _ = torch.sort(torch.rand(n_train) * 5) # torch.rand一个均匀分布，torch.randn一个是标准正态分布。
def f(x):
    return 2 * torch.sin(x) + x ** 0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train, )) # 训练样本的输出
x_test = torch.arange(0, 5, 0.1) # 测试样本
y_truth = f(x_test) # 测试样本的真实输出
n_test = len(x_test)
# 将绘制所有的训练样本（样本由圆圈表⽰），不带噪声项的真实数据⽣成函数f（标记为“Truth”），
# 以及学习得到的预测函数（标记为“Pred”）。
def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
        xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)

# 平均汇聚
# 先使⽤最简单的估计器来解决回归问题：基于平均汇聚来计算所有训练样本输出值的平均值  f = 1 /n * ∑i=1 yi
# ,
# y_hat = torch.repeat_interleave(y_train.mean(), n_test) # torch.repeat_interleave重复张量的元素
# plot_kernel_reg(y_hat)

#  f = ∑i=1 K(x − xi) /∑n,j=1 K(x − xj )yi
# 其中K是核（kernel）。公式 (10.2.3)所描述的估计器被称为 Nadaraya-Watson核回归（Nadaraya-Watson kernel
# regression）。

# 但受此启发，我们可以从 图10.1.3中的注意⼒机制框架的⻆度重写 (10.2.3)，成为⼀个更加通⽤的注意⼒汇聚（attention pooling）公式
# f(x) =n∑i=1 α(x, xi)yi
# 其中x是查询，(xi, yi)是键值对, 注意⼒汇聚是yi的加权平均
# 将查询x和键xi之间的关系建模为 注意⼒权重（attention weight）α(x, xi), 这个权重将被分配给每⼀个对应值yi
# 对于任何查询，模型在所有键值对注意⼒权重都是⼀个有效的概率分布：它们是⾮负的，并且总和为1

# 如果⼀个键xi越是接近给定的查询x，那么分配给这个键对应值yi的注意⼒权重就会越⼤，也
# 就“获得了更多的注意⼒”

# # X_repeat的形状:(n_test,n_train),
# # 每⼀⾏都包含着相同的测试输⼊（例如：同样的查询）
"""
repeat的参数是每一个维度上重复的次数， # repeat相当于将该张量复制，然后在某一维度concat起来
repeat_interleave的参数是重复的次数和维度。# 而repeat_interleave是将张量中的元素沿某一维度复制n次，即复制后的张量沿该维度相邻的n个元素是相同的
"""
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train)) # torch.repeat_interleave重复张量的元素
# x_train包含着键， attention_weights的形状：(n_test,n_train),
# 每⼀⾏都包含着要在给定的每个查询的值（y_train）之间分配的注意⼒权重
attention_weights = nn.functional.softmax(- (X_repeat - x_train) ** 2 / 2, dim=1)
# # y_hat的每个元素都是值的加权平均值，其中的权重是注意⼒权重
y_hat = torch.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)

# 现在，我们来观察注意⼒的权重。这⾥  测试数据的输⼊ 相当于查询，⽽训练数据的输⼊相当于键
# 因为两个输⼊都是经过排序的，因此由观察可知“查询-键”对越接近，注意⼒汇聚的注意⼒权重就越⾼。
d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0), xlabel='Sorted training inputs', ylabel='Sorted testing inputs')

######################################
X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))
"""
torch.bmm(input, mat2)
对 input 和 mat2 矩阵执行批处理矩阵积。input 和 mat2 必须是三维张量，每个张量包含相同数量的矩阵。
input tensor 维度：(b×n×m) ；
mat2 tensor 维度： (b×m×p) ,
"""
print(torch.bmm(X, Y).shape)

weights = torch.ones((2, 10)) * 0.1
values = torch.arange(20.0).reshape((2, 10))
print(torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1)))

# 基于 (10.2.7)中的带参数的注意⼒汇聚，使⽤⼩批量矩阵乘法，定义Nadaraya-Watson核回归的带参数版本为
#  f(x) = ∑i=1 softmax (−1/2((x − xi)w) ** 2 )yi.
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super(NWKernelRegression, self).__init__()
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(-((queries - keys) * self.w) ** 2 / 2, dim=1)
        # values的形状为(查询个数，“键－值”对个数)
        return torch.bmm(self.attention_weights.unsqueeze(1), values.unsqueeze(-1)).reshape(-1)

# 训练
"""接下来，将 训练数据集 变换为 键和值 ⽤于训练注意⼒模型。
在带参数的注意⼒汇聚模型中，任何⼀个训练样本的输⼊ 都会和 除⾃⼰以外的所有训练样本的“键－值”对进⾏计算，从⽽得到其对应的预测输出。 """
# # X_tile的形状:(n_train，n_train)，每⼀⾏都包含着相同的训练输⼊
X_tile = x_train.repeat((n_train, 1)) # 复制 n_train = 50
# # Y_tile的形状:(n_train，n_train)，每⼀⾏都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))
# keys的形状:('n_train'，'n_train'-1)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

# # values的形状:('n_train'，'n_train'-1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

# 训练带参数的注意⼒汇聚模型时，使⽤平⽅损失函数和随机梯度下降。
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])
for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch+1}, loss {float(l.sum()):.6f}')
    animator.add(epoch+1, float(l.sum()))

# 训练完带参数的注意⼒汇聚模型后，我们发现：在尝试拟合带噪声的训练数据时，预测结果绘制
# 的线不如之前⾮参数模型的平滑
# keys的形状:(n_test，n_train)，每⼀⾏包含着相同的训练输⼊（例如，相同的键）
keys = x_train.repeat((n_test, 1))
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)
# 模型加⼊可学习的参数后，曲线在注意⼒权重较⼤的区域变得更不平滑。
d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs', ylabel='Sorted testing inputs')
d2l.plt.show()








