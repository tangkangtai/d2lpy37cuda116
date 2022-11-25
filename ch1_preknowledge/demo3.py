import torch
x = torch.tensor(3.0)
y = torch.tensor(2.0)
print(x+y)
print(x-y)
print(x*y)
print(x/y)
print(x**y)

x = torch.arange(4)
print(x)
print(x[3])
print(len(x))
print(x.shape)

A = torch.arange(20).reshape(5, 4)
print(A)
print(id(A)==id(A.T))
print(A.T)

B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B)
print(B == B.T)

X = torch.arange(24).reshape(2, 3, 4)

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone() # 通过分配新内存，将A的⼀个副本分配给B
print(id(A) == id(B))  # False

# 两个矩阵的按元素乘法称为Hadamard积（Hadamard product）（数学符号⊙）
print(A * B)

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X)
print((X * a).shape)


# 降维度

x = torch.arange(4, dtype=torch.float32)
print(x.sum())

print(A)
print(A.sum(axis=0))

print(A.mean())
print(A.sum()/ A.numel())

#  非降维求和
sum_A = A.sum(axis=1, keepdims=True)
print(sum_A)
print(A / sum_A) # 利用广播机制

# 想沿某个轴计算A元素的累积总和，⽐如axis=0（按⾏计算），我们可以调⽤cumsum函数。此函数
# 不会沿任何轴降低输⼊张量的维度。
print(A)
print(A.cumsum(axis=0))
print(A.cumsum(axis=1))


# 点积
# 点积（dot product）x⊤y （或⟨x, y⟩）是相同位置的按元素乘积
# 点积表⽰加权平均（weighted average）。
# 将两个向量规范化得到单位⻓度后，点积表⽰它们夹⻆的余弦。
y = torch.ones(4, dtype=torch.float32)
x = torch.arange(4, dtype=torch.float32)
print(x)
print(y)
print(torch.dot(x, y))

print(torch.sum(x*y))

# 解矩阵-向量积

# A的列维数（沿轴1的⻓度）必须与x的维数（其⻓度）相同
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
x = torch.arange(4, dtype=torch.float32)
print(A.shape)
print(x.shape)
print(torch.mv(A, x))


# 矩阵乘法
B = torch.ones(4, 3)
print(torch.mm(A, B))

# L2范数 欧⼏⾥得距离是⼀个L2范数, 其L2范数是向量元素平⽅和的平⽅根
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))

# ，L1范数受异常值的影响较⼩。为了计算L1范数，我们将绝对值函数和按元素求和组合起来。
print(torch.abs(u).sum())

# Frobenius范数（Frobenius norm）是矩阵元素平⽅和的平⽅根：
print(torch.norm(torch.ones((4, 9))))


