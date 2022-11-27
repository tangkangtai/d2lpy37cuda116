import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

""":微调（fine-tuning）
# 1. 在源数据集（例如ImageNet数据集）上预训练神经⽹络模型，即源模型。
# 2. 创建⼀个新的神经⽹络模型，即⽬标模型。这将复制源模型上的所有模型设计及其参数（输出层除外）。
#       我们假定这些模型参数包含从源数据集中学到的知识，这些知识也将适⽤于⽬标数据集。我们还假设
#       源模型的输出层与源数据集的标签密切相关；因此不在⽬标模型中使⽤该层。
# 3. 向⽬标模型添加输出层，其输出数是⽬标数据集中的类别数。然后随机初始化该层的模型参数。
# 4. 在⽬标数据集（如椅⼦数据集）上训练⽬标模型。输出层将从头开始进⾏训练，⽽所有其他层的参数将
#       根据源模型的参数进⾏微调。"""
# ###################### 热狗识别 ################
# 我们将在⼀个⼩型数据集上微调ResNet模型。该模型已在ImageNet数据集上进⾏了预训练。
# 这个⼩型数据集包含数千张包含热狗和不包含热狗的图像，我们将使⽤微调模型来识别图像中是否包含热狗。

# 获取数据集
# 我们使⽤的热狗数据集来源于⽹络。该数据集包含1400张热狗的“正类”图像，以及包含尽可能多的其他⻝
# 物的“负类”图像。含着两个类别的1000张图⽚⽤于训练，其余的则⽤于测试
# 解压下载的数据集，我们获得了两个⽂件夹hotdog/train和hotdog/test。这两个⽂件夹都有hotdog
# （有热狗）和not-hotdog（⽆热狗）两个⼦⽂件夹，⼦⽂件夹内都包含相应类的图像
# d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 'fba480ffa8aa7e0febbb511d181409f899b9baa5')
# data_dir = d2l.download_extract('hotdog')

# 我们创建两个实例来分别读取训练和测试数据集中的所有图像⽂件
train_imgs = torchvision.datasets.ImageFolder(os.path.join('../data/hotdog', 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join('../data/hotdog', 'test'))

# 下⾯显⽰了前8个正类样本图⽚和最后8张负类样本图⽚。正如你所看到的，图像的⼤⼩和纵横⽐各有不同。
# hotdogs = [train_imgs[i][0] for i in range(8)]
# not_hotdogs = [train_imgs[-i-1][0] for i in range(8)]
# d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
# d2l.plt.show()
# print(train_imgs[0]) # PIL图片
# print(test_imgs)# ImageFolder对象


# 在训练期间，我们⾸先从图像中裁切随机⼤⼩和随机⻓宽⽐的区域，然后将该区域缩放为224×224输⼊图像。

# 在测试过程中，我们将图像的⾼度和宽度都缩放到256像素，然后裁剪中央224 × 224区域作为输⼊。
# 此外，对于RGB（红、绿和蓝）颜⾊通道，我们分别标准化每个通道。
# 具体⽽⾔，该通道的每个值减去该通道的平均值，然后将结果除以该通道的标准差。

normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize
])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])
# 定义和初始化模型
# 我们使⽤在ImageNet数据集上预训练的ResNet-18作为源模型。
# 在这⾥，我们指定pretrained=True以⾃动下载预训练的模型参数。
# 如果你⾸次使⽤此模型，则需要连接互联⽹才能下载。

# pretrained_net = torchvision.models.resnet18(pretrained=True)

# 预训练的源模型实例包含许多特征层和⼀个输出层fc。
# 此划分的主要⽬的是促进对除输出层以外所有层的模型参数进⾏微调。下⾯给出了源模型的成员变量fc。
# print(pretrained_net.fc)

# 在ResNet的全局平均汇聚层后，全连接层转换为ImageNet数据集的1000个类输出。
# 之后，我们构建⼀个新的神经⽹络作为⽬标模型。
# 它的定义⽅式与预训练源模型的定义⽅式相同，只是最终层中的输出数量被设置⽬标数据集中的类数（⽽不是1000个）


###############
# ⽬标模型finetune_net中成员变量features的参数被初始化为源模型相应层的模型参
# 数。由于模型参数是在ImageNet数据集上预训练的，并且⾜够好，因此通常只需要较⼩的学习率即可微调这
# 些参数。
# 成员变量output的参数是随机初始化的，通常需要更⾼的学习率才能从头开始训练。假设Trainer实例中
# 的学习率为η，我们将成员变量output中参数的学习率设置为10η。
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)

# 定义了⼀个训练函数train_fine_tuning，该函数使⽤微调，因此可以多次调⽤
# # 如果param_group=True，输出层中的模型参数将使⽤⼗倍的学习率
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join('../data/hotdog', 'train'), transform=train_augs), batch_size=batch_size, shuffle=True)

    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join('../data/hotdog', 'test'), transform=test_augs), batch_size=batch_size)

    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction='none')
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
                     if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(), 'lr': learning_rate * 10}],
                                  lr=learning_rate, weight_decay=0.001)

    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)

    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

# 我们使⽤较⼩的学习率，通过微调预训练获得的模型参数。
train_fine_tuning(finetune_net, 5e-5)
################################################
# 为了进⾏⽐较，我们定义了⼀个相同的模型，但是将其所有模型参数初始化为随机值。由于整个模型需要从
# 头开始训练，因此我们需要使⽤更⼤的学习率
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)

# 意料之中，微调模型往往表现更好，因为它的初始参数值更有效



