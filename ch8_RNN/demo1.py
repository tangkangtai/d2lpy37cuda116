import torch
from torch import nn
from d2l import torch as d2l

# 处理序列数据需要统计⼯具和新的深度神经⽹络架构
# 第⼀种策略，假设在现实情况下相当⻓的序列 xt−1, . . . , x1可能是不必要的，因此我们只需要满⾜某个⻓度
# 为τ的时间跨度，即使⽤观测序列xt−1, . . . , xt−τ。
# 当下获得的最直接的好处就是参数的数量总是不变的，⾄少在t > τ时如此，这就使我们能够训练⼀个上⾯提及的深度⽹络。
# 这种模型被称为⾃回归模型（autoregressive models），因为它们是对⾃⼰执⾏回归

# 第⼆种策略，如 图8.1.2所⽰，是保留⼀些对过去观测的总结ht，并且同时更新预测xˆt和总结ht。这就产⽣了
# 基于xˆt = P(xt | ht)估计xt，以及公式ht = g(ht−1, xt−1)更新的模型。由于ht从未被观测到，这类模型也被称
# 为 隐变量⾃回归模型（latent autoregressive models）。
T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
# 我们将这个序列转换为模型的“特征－标签”（feature-label）对。基于嵌⼊维度τ，我们将数据映射
# 为数据对yt = xt 和xt = [xt−τ , . . . , xt−1]。
# 如果拥有⾜够⻓的序列就丢弃这⼏项；
# 另⼀个⽅法是⽤零填充序列。在这⾥，我们仅使⽤前600个“特征－标签”对进⾏训练
tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i:T-tau+i]
labels = x[tau:].reshape((-1, 1))
batch_size, n_train = 16, 600
# 只有前n_train个样本⽤于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)
# 使⽤⼀个相当简单的架构训练模型：⼀个拥有两个全连接层的多层感知机，ReLU激活函数和平⽅损失

# 初始化⽹络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
# ⼀个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
    nn.ReLU(),
    nn.Linear(10, 1))
    net.apply(init_weights)
    return net
# 平⽅损失。注意：MSELoss计算平⽅误差时不带系数1/2
loss = nn.MSELoss(reduction='none')

def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
    print(f'epoch {epoch + 1}, 'f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')
net = get_net()
train(net, train_iter, loss, 5, 0.01)

onestep_preds = net(features)
d2l.plot([time, time[tau:]], [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3))