import torch


print(torch.device('cpu'))
print(torch.cuda.device('cuda'))
# 查看可用gpu数量
print(torch.cuda.device_count())
print(torch.__version__)
print(torch.cuda.is_available())
a = torch.ones((3, 1))
a = a.cuda(0)

b = torch.ones((3, 1))
b = b.cuda(0)
c = a + b
print(c)