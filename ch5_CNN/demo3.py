
import torch
from torch import nn
# 假设输⼊形状为nh × nw，卷积核形状为kh × kw，
# 那么输出形状将是(nh − kh + 1) × (nw − kw + 1)。因此，卷积的输出形状取决于输⼊形状和卷积核的形状。
#################
# 需要设置ph = kh − 1 # 和pw = kw − 1
### 假设kh是奇数，我们将在⾼度的两侧填充ph/2⾏
### 果kh是偶数，则⼀种可能性是在输⼊顶部填充⌈ph/2⌉⾏，在底部填充⌊ph/2⌋⾏

# 为了⽅便起⻅，我们定义了⼀个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输⼊和输出提⾼和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这⾥的（1，1）表⽰批量⼤⼩和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量⼤⼩和通道
    return Y.reshape(Y.shape[2:])

# 请注意，这⾥每边都填充了1⾏或1列，因此总共添加了2⾏或2列
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size = (8, 8))
print(comp_conv2d(conv2d, X).shape)

# 当卷积核的⾼度和宽度不同时，我们可以填充不同的⾼度和宽度，使输出和输⼊具有相同的⾼度和宽度
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)
