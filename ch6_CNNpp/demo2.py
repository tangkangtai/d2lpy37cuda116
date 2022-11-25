


######### VGG块#####################
# ⼀个VGG块与之类似，由⼀系列卷积层组成，后⾯再加上⽤于空间下采样的最⼤汇聚层。
# VGG论⽂中 [Simonyan & Zisserman, 2014]，作者使⽤了带有3 × 3卷积核、填充为1（保持⾼度和宽度）的卷积层，
# 和带有2 × 2汇聚窗⼝、步幅为2（每个块后的分辨率减半）的最⼤汇聚层
import torch
from torch import nn
from d2l import torch as d2l

# 卷积层的数量num_convs、输⼊通道的数量in_channels 和输出通道的数量out_channels
def vgg_block(num_convs, in_channels, out_channels):
    layer = []
    for _ in range(num_convs):
        layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layer.append(nn.ReLU())
        in_channels = out_channels

    layer.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layer)

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
# 代码实现了VGG-11。可以通过在conv_arch上执⾏for循环来简单实现。
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )
# net = vgg(conv_arch)


X = torch.randn(size=(1, 1, 224, 224))
# for blk in net:
#     X = blk(X)
#     print(blk.__class__.__name__, 'output shape:\t', X.shape)
#
# # 由于VGG-11⽐AlexNet计算量更⼤，因此我们构建了⼀个通道数较少的⽹络，⾜够⽤于训练Fashion-MNIST数据集
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
