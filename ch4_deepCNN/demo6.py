import torch
from torch import nn

torch.device('cpu')
torch.device('cuda')
torch.device('cuda:1')

print(torch.cuda.device_count())

def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """返回所有可⽤的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

# 张量与GPU
x = torch.tensor([1, 2, 3])
print(x.device)

X = torch.ones(2, 3, device=try_gpu())
print(X)

# 复制
# 要计算X + Y，我们需要决定在哪⾥执⾏这个操作
Z = X.cuda(0)
print(X)
print(Z)

net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
print(net(X))
print(net[0].weight.data.device)