import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型参数
num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# 实现softmax由三个步骤组成：
# 1. 对每个项求幂（使⽤exp）；
# 2. 对每⼀⾏求和（⼩批量中每个样本是⼀⾏），得到每个样本的规范化常数；
# 3. 将每⼀⾏除以其规范化常数，确保结果的和为1。

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这⾥应⽤了⼴播机制

# 对于任何随机输⼊，我们将每个元素变成⼀个⾮负数。此外，依据概率原理，每⾏总和为1
# X = torch.normal(0, 1, (2, 5))
# # print(X)
# X_prob = softmax(X)
# print(X_prob)
# print(X_prob.sum(1))

# 定义模型
# 将数据传递到模型之前，我们使⽤reshape函数将每张原始图像展平为向量。
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

# 只需⼀⾏代码就可以实现交叉熵损失函数。
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])


y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
cross_entropy(y_hat, y)

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# 第⼀个样本的预测类别是2（该⾏的最⼤元素为0.6，索引为2），这与实际标签0不⼀致。
# 第⼆个样本的预测类别是2（该⾏的最⼤元素为0.5，索引为2），这与实际标签2⼀致。
# 因此，这两个样本的分类精度率为0.5。
# print(accuracy(y_hat, y) / len(y))
############################
# print(y_hat.shape)
# print(y_hat.argmax(axis=1))
# print(y_hat.argmax(axis=0))
# print(y.dtype)
# print(y_hat)
# y_hat = y_hat.argmax(axis=1)
# print(y_hat)
# print(y_hat.type(y.dtype))
# print(y_hat.type(y.dtype) == y)
# print((y_hat.type(y.dtype) == y).type(y.dtype).sum())
####################

# 评估在任意模型net的精度。
# 在Accumulator实例中创建了2个变量，分别⽤于存储正确预测的数量和预测的总数量。当我们遍
# 历数据集时，两者都将随着时间的推移⽽累加。
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, * args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval() # 将模型设置为评估模式

    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y ), y.numel())
    return metric[0] / metric[1]

# l = [1, 2]
# p = [3, 4]
# for i, j in zip(l, p):
#     print(i,",", j)

print(evaluate_accuracy(net,test_iter))

# 训练
# 训练
def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型⼀个迭代周期（定义⻅第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Sequential):
        net.eval()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使⽤PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使⽤定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

# 正如基本的str,tuple,list,dict,set等，它们可以使用len()函数，也仅仅是因为它们的类实现了__len__函数而已
class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear',yscale='linear',
                 fmts=('-','m--','g-.','r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]
        # 使⽤lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n

        if not self.X:
            self.X = [[] for _ in range(n)]

        if not self.Y:
            self.Y = [[] for _ in range(n)]

        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        d2l.plt.show()
        display.display(self.fig)       # from IPython import display
        display.clear_output(wait=True)


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9], legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metric = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metric + (test_acc,))

    d2l.plt.show()
    train_loss, train_acc = train_metric
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1, train_acc
    assert test_acc <= 1 and test_acc>0.7, test_acc

lr = 0.1
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

# 预测
def predict_ch3(net, test_iter, n=6):
    """预测标签（定义⻅第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n,28,28)), 1, n, titles=titles[0:n])
    d2l.plt.show()

predict_ch3(net, test_iter)