import torch
import numpy as np
# x = torch.arange(12)
# print(x)
# print(x.shape)
# print(x.numel())  #  张量中元素的总数，即形状的所有元素乘积，可以检查它的⼤⼩（size）
#######################################################
# pytorch的resize_可以改变tensor形状(任意形状，少的补，多的去掉),会改变原tensor的形状，
# 但是原tensor的元素貌似会保留，之后改变回原形状，原来的元素还在
# y = torch.arange(24).reshape(4, 6)
# print(y.numel())
# print(y.shape)
# y1 = y.reshape(3, 8)
# # print(y)
# print(y1.shape)
# print(y.shape)
# y1 = y.reshape(3,8)
# print(y1.shape)
# print(y.shape)
# y = y.resize_(2,3)
# print(y)
# print(y.resize_(3,9))
##############################
#  调用numpy的 resize, np.resize(A, (x , y)) 也可以改变数组形状
# 直接 A.resize(x, y) 也可改变原数组的大小， 多的去掉， 烧的补0
# n1 = np.arange(10)
# # print(n1.shape)
# #
# n2 = n1.reshape(2,5)
# print(n2.shape)

# # n3 = n1.resize(2, 4)          # error
# # print(n3)
# # print(n3.shape)
#
# n1.resize(2,3)
# print(n1)
# n1.resize(2, 5)
# print(n1)
# n3 = np.resize(n1, (2, 3))
# print(n3)
# print(n3.shape)
# print(n1.shape)
# n1 = np.resize(n1, (2, 4))
# print(n1.shape)
# print(n1)
# n1 = np.resize(n1, (2, 5))
# print(n1.shape)
# print(n1)
###########################

# x = np.random.randint(100)
# print(x)
# y = np.arange(1000).reshape(10, 10, 10)
# print(y.shape)
# print(np.prod(list(y.shape)))
#
# a = np.arange(6)
# b = np.prod(a[1:])
# print(b)
#
# a1 = np.arange(1, 13).reshape(3, 4)
# print(a1)
# # print(a1[0])
# print(np.prod(a1, axis=0))  # 求数组各个维度的元素相乘的值
# print(np.prod(a1, axis=1))
#############全0   全1 张量#############################
# zeros = torch.zeros((2, 3, 4))
# print(zeros)
# print(zeros.shape)
#
# ones = torch.ones((2, 3, 4))
# print(ones)
# print(ones.shape)
#
# x = torch.arange(36).reshape(2,3, 6)
# x_ones = torch.ones_like(x)
# x_zeros = torch.zeros_like(x)
# print(x_ones)
# print(x_ones.shape)
# print(x_zeros.shape)

####################高斯分布采样的数据##############
# r = torch.randn(3, 4) # (3, 4)   每个元素都从均值为0、标准差为1的标准⾼斯分布
# print(r)

# p = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# print(p)
# print(p.shape)
# print(torch.argmax(p, axis=0))  # 返回指定维度的最大值的indice  axis=0每列最大
# print(torch.argmax(p, axis=1))  # 返回指定维度的最大值的indice  axis=1每行最大


########### tensor 的加减乘除
# x = torch.tensor([1.0, 2, 4, 8])
# y = torch.tensor([2,2 ,2, 2])
# print(x+y)
# print(x-y)
# print(x*y)
# print(x/y)
# print(x**y)
# print(torch.exp(x))

# 张量连接
# X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
# print(X)
Y = torch.tensor([[2.0,1,4,3], [1, 2,3,4], [4,3,2,1]])
# print(Y.shape)
# print(torch.cat((X, Y), dim=0))
# print(torch.cat((X, Y), dim=1))

# print(torch.sum(Y))
#
# print(X == Y)
#
# print(X.sum())
# print(Y.sum())
# print(X.mean())
# print(torch.mean(Y, axis=1))
# print(torch.prod(Y))
# print(torch.prod(Y, axis=0))
# print(torch.prod(Y, axis=1))
###########################广播机制########
# a = torch.arange(3).reshape((3, 1))
# b = torch.arange(2).reshape((1, 2))
# print(a)
# print(b)
# print(a+b)
# print()

###########################################
# X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
# print(X)
# print(X[1:3])  # 切片第一个会选择， 最后一个不会选择
#
# X[1:2, :] = 10  # 切片写入

# 内存演示
# before = id(X)
# X = X + Y
# print(id(X) == before)
#
# Z = torch.zeros_like(Y)
# print('id(Z):', id(Z))
# Z[:] = X + Y
# print('id(Z):', id(Z))

# 后续计算中没有重复使⽤X，我们也可以使⽤X[:] = X + Y或X += Y来减少操作的内存开销

# before = id(X)
# X += Y
# print(id(X) == before)

#################张量与numpy互转########################
# X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
# print(X)
# A = X.numpy()
# print(type(A))
#
# B = torch.tensor(A)
# print(type(B))

#将⼤⼩为1的张量转换为Python标量，我们可以调⽤item函数或Python的内置函数。

# a = torch.tensor([3.5])
# print(a)
# print(a.item())
# print(float(a))
# print(int(a))

X = torch.arange(24).reshape(2,3,4) #
print(X)
# b1 = torch.arange(6).reshape(2,3,2)
b2 = torch.arange(8).reshape(2,1,4)
b3 = torch.arange(12).reshape(1,3,4)
# print(b)
# print(X+b1)
print(X+b2)
print(X+b3)