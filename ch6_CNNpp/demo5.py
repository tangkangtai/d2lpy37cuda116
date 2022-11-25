import torch
from torch import nn
from d2l import torch as d2l

############# 批量规范化层#################
# 对于卷积层，我们可以在卷积层之后和⾮线性激活函数之前应⽤批量规范化。当卷积有多个输出通道
# 时，我们需要对这些通道的“每个”输出执⾏批量规范化，每个通道都有⾃⼰的拉伸（scale）和偏移（shift）
# 参数，这两个参数都是标量。假设我们的⼩批量包含m个样本，并且对于每个通道，卷积的输出具有⾼度p和
# 宽度q。那么对于卷积层，我们在每个输出通道的m · p · q个元素上同时执⾏每个批量规范化。

# 3 从零实现
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使⽤传⼊的移动平均所得的均值和⽅差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert  len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使⽤全连接层的情况，计算特征维上的均值和⽅差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使⽤⼆维卷积层的情况，计算通道维上（axis=1）的均值和⽅差。
            # 这⾥我们需要保持X的形状以便后⾯可以做⼴播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，⽤当前的均值和⽅差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和⽅差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta # 缩放和移位
    return Y, moving_mean.data, moving_var.data

# 以创建⼀个正确的BatchNorm层。这个层将保持适当的参数：拉伸gamma和偏移beta,这两个参
# 数将在训练过程中更新。此外，我们的层将保存均值和⽅差的移动平均值，以便在模型预测期间随后使⽤。
class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表⽰完全连接层，4表⽰卷积层
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)

        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beat = nn.Parameter(torch.zeros(shape))
        # ⾮模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)

        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return Y
