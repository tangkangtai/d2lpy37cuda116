

import torch
from torch import nn
from d2l import torch as d2l
# 稠密连接⽹络（DenseNet）
# DenseNet使⽤了ResNet改良版的“批量规范化、激活和卷积”架构
def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
    )

# ⼀个稠密块由多个卷积块组成，每个卷积块使⽤相同数量的输出通道
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))

        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输⼊和输出
            X = torch.cat((X, Y), dim=1)
        return X
# 我们定义⼀个有2个输出通道数为10的DenseBlock。
# 使⽤通道数为3的输⼊时，我们会得到通道数为3 + 2 × 10 = 23的输出。
# 卷积块的通道数控制了输出通道数相对于输⼊通道数的增⻓，因此也被
# 称为增⻓率（growth rate）。
blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
print(Y.shape)

# 由于每个稠密块都会带来通道数的增加，使⽤过多则会过于复杂化模型。⽽过渡层可以⽤来控制模型复杂度。
# 它通过1 × 1卷积层来减⼩通道数，并使⽤步幅为2的平均汇聚层减半⾼和宽，从⽽进⼀步降低模型复杂度。
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))

blk = transition_block(23, 10)
print(blk(Y).shape)

# DenseNet模型
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
# 似于ResNet使⽤的4个残差块，DenseNet使⽤的是4个稠密块。与ResNet类似，我们可以设置每个
# 稠密块使⽤多少个卷积层。这⾥我们设成4，从⽽与 7.6节的ResNet-18保持⼀致。稠密块⾥的卷积层通道数
# （即增⻓率）设为32，所以每个稠密块将增加128个通道
# # num_channels为当前的通道数
num_channels, growth_rate = 64, 32
num_conv_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_conv_in_dense_blocks):
    blk.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 上⼀个稠密块的输出通道数
    num_channels += num_convs * growth_rate
    # 在稠密块之间添加⼀个转换层，使通道数量减半
    if i != len(num_conv_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

# 与ResNet类似，最后接上全局汇聚层和全连接层来输出结果
net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10)
)
