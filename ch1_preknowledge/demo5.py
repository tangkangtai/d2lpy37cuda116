import torch
x = torch.arange(4.0)
print(x)

"""
在我们计算y关于x的梯度之前，我们需要⼀个地⽅来存储梯度。重要的是，我们不会在每次对⼀个参数求导
时都分配新的内存。因为我们经常会成千上万次地更新相同的参数，每次都分配新的内存可能很快就会将内
存耗尽
"""
print(x.requires_grad_(True)) # 等价于x=torch.arange(4.0,requires_grad=True)
print(x.grad) # 默认值是None


# x是⼀个⻓度为4的向量，计算x和x的点积，得到了我们赋值给y的标量输出
y = 2 * torch.dot(x, x)
# print(y)
# # 我们通过调⽤反向传播函数来⾃动计算y关于x每个分量的梯度，并打印这些梯度。
y.backward()
print(x.grad)
print(x.grad == 4 * x)
#
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

# ⾮标量变量的反向传播
"""
虽然这些更奇特的对象确实出现在⾼级机器学习中（包括深度学习中），但当我们调⽤向量的反向计算
时，我们通常会试图计算⼀批训练样本中每个组成部分的损失函数的导数。这⾥，我们的⽬的不是计算微分
矩阵，⽽是单独计算批量中每个样本的偏导数之和。
"""
x.grad.zero_()
y = x * x
print(y)
print(y.sum())
y.sum().backward() # 等价于y.backward(torch.ones(len(x)))
print(x.grad)


# 我们可以分离y来返回⼀个新变量u，该变量与y具有相同的值，但丢弃计算图中如何计算y的任何信
# 息。换句话说，梯度不会向后流经u到x。因此，下⾯的反向传播函数计算z=u*x关于x的偏导数，同时将u作
# 为常数处理，⽽不是z=x*x*x关于x的偏导数。
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad == u)
