import torch
from torch import nn
from d2l import torch as d2l

# AlexNet
# AlexNet通过暂退法（4.6节）控制全连接层的模型复杂度
# AlexNet在训练时增加了⼤量的图像增强数据，如翻转、裁切和变⾊。这使得模型更健壮，更⼤的样本量
# 有效地减少了过拟合
net = nn.Sequential(
    # 这⾥，我们使⽤⼀个11*11的更⼤窗⼝来捕捉对象。
    # 同时，步幅为4，以减少输出的⾼度和宽度。
    # 另外，输出通道的数⽬远⼤于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride = 4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减⼩卷积窗⼝，使⽤填充为2来使得输⼊与输出的⾼和宽⼀致，且增⼤输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使⽤三个连续的卷积层和较⼩的卷积窗⼝。
    # 除了最后的卷积层，输出通道的数量进⼀步增加。
    # 在前两个卷积层之后，汇聚层不⽤于减少输⼊的⾼度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这⾥，全连接层的输出数量是LeNet中的好⼏倍。使⽤dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这⾥使⽤Fashion-MNIST，所以⽤类别数为10，⽽⾮论⽂中的1000
    nn.Linear(4096, 10))

# 构造⼀个⾼度和宽度都为224的单通道数据，来观察每⼀层输出的形状
###########################
# 假设输⼊形状为nh × nw，卷积核形状为kh × kw，
# 那么输出形状将是(nh − kh + 1) × (nw − kw + 1)。因此，卷积的输出形状取决于输⼊形状和卷积核的形状。
#################
# 需要设置ph = kh − 1 # 和pw = kw − 1
### 假设kh是奇数，我们将在⾼度的两侧填充ph/2⾏
### 果kh是偶数，则⼀种可能性是在输⼊顶部填充⌈ph/2⌉⾏，在底部填充⌊ph/2⌋⾏

# ⌊(nh − kh + ph + sh) / sh⌋ × ⌊(nw − kw + pw + sw) / sw⌋.
##########################
# X = torch.randn(1, 1, 224, 224)
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape:\t', X.shape)


batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


