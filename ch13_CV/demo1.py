import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# 图像增广
# 在对常⽤图像增⼴⽅法的探索时，我们将使⽤下⾯这个尺⼨为400 × 500的图像作为⽰例。
d2l.set_figsize()
img = d2l.Image.open('image/cat1.jpg')
# print(img)# shape = 540, 360
# d2l.plt.imshow(img)
# d2l.plt.show()

# ⼤多数图像增⼴⽅法都具有⼀定的随机性。为了便于观察图像增⼴的效果，我们下⾯定义辅助函数apply。
# 此函数在输⼊图像img上多次运⾏图像增⼴⽅法aug并显⽰所有结果
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
    d2l.plt.show()

###################### 翻转和裁剪###################
# 左右翻转图像通常不会改变对象的类别。这是最早且最⼴泛使⽤的图像增⼴⽅法之⼀。
# 接下来，我们使⽤transforms模块来创建RandomFlipLeftRight实例，这样就各有50%的⼏率使图像向左或向右翻转。
# apply(img, torchvision.transforms.RandomHorizontalFlip())

# 上下翻转图像不如左右图像翻转那样常⽤。但是，⾄少对于这个⽰例图像，上下翻转不会妨碍识别。接下来，
# 我们创建⼀个RandomFlipTopBottom实例，使图像各有50%的⼏率向上或向下翻转
# apply(img, torchvision.transforms.RandomVerticalFlip())

# 我们可以通过对图像进⾏随机裁剪，使物体以不同的⽐例出现在图像的不同位置。
# 这也可以降低模型对⽬标位置的敏感性。
# 在下⾯的代码中，我们随机裁剪⼀个⾯积为原始⾯积10%到100%的区域，该区域的宽⾼⽐从0.5到2之间随机取值。
# 然后，区域的宽度和⾼度都被缩放到200像素。
# 在本节中（除⾮另有说明），a和b之间的随机数指的是在区间[a, b]中通过均匀采样获得的连续值。

shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
# apply(img, shape_aug)

#################改变颜色########################
# 我们可以改变图像颜⾊的四个⽅⾯：亮度、对⽐度、饱和度和⾊调。在下⾯的
# ⽰例中，我们随机更改图像的亮度，随机值为原始图像的50%（1 − 0.5）到150%（1 + 0.5）之间。
# apply(img, torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0))

# 我们可以随机更改图像的⾊调
# apply(img, torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5))

# 我们还可以创建⼀个RandomColorJitter实例，并设置如何同时随机更改图像的亮度（brightness）、对
# ⽐度（contrast）、饱和度（saturation）和⾊调（hue）。
color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
# apply(img, color_aug)


# 结合多种图像增⼴⽅法
augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
# apply(img, augs)

# 使⽤图像增⼴进⾏训练
# 让我们使⽤图像增⼴来训练模型。这⾥，我们使⽤CIFAR-10数据集，⽽不是我们之前使⽤的Fashion-MNIST数
# 据集。这是因为Fashion-MNIST数据集中对象的位置和⼤⼩已被规范化，⽽CIFAR-10数据集中对象的颜⾊和
# ⼤⼩差异更明显。CIFAR-10数据集中的前32个训练图像如下所⽰
all_images = torchvision.datasets.CIFAR10(train=True, root="../data", download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)

# 我们通常对训练样本只进⾏图像增⼴，且在预测过程中不使⽤随机操作的图像增⼴
# 我们只使⽤最简单的随机左右翻转。此外，我们使⽤ToTensor实例将⼀批图像转换
# 为深度学习框架所要求的格式，即形状为（批量⼤⼩，通道数，⾼度，宽度）的32位浮点数，取值范围为0到1。
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# 接下来，我们定义⼀个辅助函数，以便于读取图像和应⽤图像增⼴。PyTorch数据集提供的transform参数
# 应⽤图像增⼴来转化图像。
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train, transform=augs, download=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train,
                                             num_workers=d2l.get_dataloader_workers())
    return dataloader

# 我们在CIFAR-10数据集上训练 7.6节中的ResNet-18模型
# 多GPU训练
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """⽤多GPU进⾏⼩批量训练"""
    if isinstance(X, list):
        # 微调BERT中所需（稍后讨论)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices=d2l.try_all_gpus()):
    """⽤多GPU进⾏模型训练"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # 4个维度：储存训练损失，训练准确度，实例数，特点数
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')

# 现在，我们可以定义train_with_data_aug函数，使⽤图像增⼴来训练模型。该函数获取所有的GPU，
# 并使⽤Adam作为训练的优化算法，将图像增⼴应⽤于训练集，最后调⽤刚刚定义的⽤于训练和评估模型
# 的train_ch13函数。
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)
def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net,train_iter, test_iter, loss, trainer, 10, devices)

train_with_data_aug(train_augs, test_augs, net)

