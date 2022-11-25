import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0到1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root='../data',train=True, transform=trans, download=True
)
mnist_test = torchvision.datasets.FashionMNIST(
    root='../data', train=True, transform=trans, download=True
)
# Fashion-MNIST由10个类别的图像组成，每个类别由训练数据集（train dataset）中的6000张图像和测试数据
# 集（test dataset）中的1000张图像组成。因此，训练集和测试集分别包含60000和10000张图像。测试数据集
# 不会⽤于训练，只⽤于评估模型性能
print(len(mnist_train))
print(len(mnist_test))

# 每个输⼊图像的⾼度和宽度均为28像素。数据集由灰度图像组成，其通道数为1。
print(mnist_train[0][0].shape)

# Fashion-MNIST中包含的10个类别，分别为t-shirt（T恤）、trouser（裤⼦）、pullover（套衫）、dress（连⾐
# 裙）、coat（外套）、sandal（凉鞋）、shirt（衬衫）、sneaker（运动鞋）、bag（包）和ankle boot（短靴）
# def get_fashion_mnist_labels(labels):
#     """返回Fashion-MNIST数据集的⽂本标签"""
#     text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
#                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
#     return [text_labels[int(i)] for i in labels]
#
# def show_images(imgs, num_rows, num_clos, titles=None, scale=1.5):
#     """绘制图像列表"""
#     figsize = (num_clos * scale, num_rows * scale)
#     _, axes = d2l.plt.subplots(num_rows, num_clos, figsize=figsize)
#     axes = axes.flatten()
#     for i, (ax, img) in enumerate(zip(axes, imgs)):
#         if torch.is_tensor(img):
#             # 图⽚张量
#             ax.imshow(img.numpy())
#         else:
#             # PIL图⽚
#             ax.imshow(img)
#         ax.axes.get_xaxis().set_visible(False)
#         ax.axes.get_yaxis().set_visible(False)
#
#         if titles:
#             ax.set_title(titles[i])
#     d2l.plt.show()
#     return axes
# # 以下是训练数据集中前⼏个样本的图像及其相应的标签。
#
# X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
#
batch_size = 256
def get_dataloader_workers():
    """使⽤4个进程来读取数据"""
    return 0

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())

timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')

