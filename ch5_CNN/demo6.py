
import torch
from torch import nn
from d2l import torch as d2l
# LeNet 它是最早发布的卷积神经⽹络之⼀，因其在计算机视觉任务中的⾼效性能⽽受到⼴泛关注
# 总体来看，LeNet（LeNet-5）由两个部分组成：
#   • 卷积编码器：由两个卷积层组成;
#   • 全连接层密集块：由三个全连接层组成。

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)

"""
在整个卷积块中，与上⼀层相⽐，每⼀层特征的⾼度和宽度都减⼩了。第⼀个卷积层使⽤2个像素的
填充，来补偿5 × 5卷积核导致的特征减少。相反，第⼆个卷积层没有填充，因此⾼度和宽度都减少了4个像
素。随着层叠的上升，通道的数量从输⼊时的1个，增加到第⼀个卷积层之后的6个，再到第⼆个卷积层之后的16个。
同时，每个汇聚层的⾼度和宽度都减半。最后，每个全连接层减少维数，最终输出⼀个维数与结果
分类数相匹配的输出。
"""
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
# 为了进⾏评估，我们需要对 3.6节中描述的evaluate_accuracy函数进⾏轻微的修改
# 由于完整的数据集位于内存中，因此在模型使⽤GPU计算数据集之前，我们需要将其复制到显存中。
def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使⽤GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device

    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_ch6(net, train_net, test_iter, num_epochs, lr, device):
    """⽤GPU训练模型(在第六章定义)"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr = lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0],d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if(i + 1) %(num_batches // 5) == 0 or i == num_batches -1:
                animator.add(epoch + (i + 1) / num_batches,(train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    d2l.plt.show()
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
              f'on {str(device)}')

lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()