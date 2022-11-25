
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
# ResNet
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
# ⼀种是当use_1x1conv=False时，应⽤ReLU⾮线性函数之前，将输⼊添加到输出。
# 另⼀种是当use_1x1conv=True时，添加通过1 × 1卷积调整通道和分辨率
"""查看输⼊和输出形状⼀致的情况"""
blk = Residual(3, 3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
print(Y.shape)

"""也可以在增加输出通道数的同时，减半输出的⾼和宽"""
blk = Residual(3, 6, use_1x1conv=True, strides=2)
print(blk(X).shape)
