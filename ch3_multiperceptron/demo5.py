import torch
from torch import nn
from d2l import torch as d2l

"""
y = 0.05 +d∑i=10.01xi + ϵ where ϵ ∼ N (0, 0.012)
"""

# 标签同时被均值为0，标准差为0.01⾼斯噪声破坏。为了使过拟合的效
# 果更加明显，我们可以将问题的维数增加到d = 200，并使⽤⼀个只包含20个样本的⼩训练集。
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5

true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)

test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

# 从头开始实现权重衰减，只需将L2的平⽅惩罚添加到原始⽬标函数中。

# 定义⼀个函数来随机初始化模型参数。
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

# 这⼀惩罚最⽅便的⽅法是对所有项求平⽅后并将它们求和。
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])

    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # ⼴播机制使l2_penalty(w)成为⼀个⻓度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w,b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))

    print('w的L2范数是：', torch.norm(w).item())

# ⽤lambd = 0禁⽤权重衰减后运⾏这个代码
train(lambd=0)
# 使⽤权重衰减来运⾏代码
train(lambd=3)

# 简洁实现
# 实例化优化器时直接通过weight_decay指定weight decay超参数。默认情况下，
# PyTorch同时衰减权重和偏移。这⾥我们只为权重设置了weight_decay，所以偏置参数b不会衰减。
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([
        {"params": net[0].weight, 'weight_decay':wd},
        {"params": net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs',ylabel='loss', yscale='log', xlim=[5, num_epochs], legend=['train','test'])
    for epoch in range(num_epochs):
        for X, y in trainer:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            train.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (d2l.evaluate_loss(net, train_iter, loss),
                          d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())