# 概率论
import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones([6]) / 6
# print(multinomial.Multinomial(1, fair_probs).sample())
# print(multinomial.Multinomial(10, fair_probs).sample())

count = multinomial.Multinomial(1000, fair_probs).sample()
# print(count / 1000)

# 我们进⾏500组实验，每组抽取10个样本
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
# print(counts.shape)

cum_counts = counts.cumsum(dim=0)
print(cum_counts.shape)
# print( cum_counts.sum(dim=1, keepdims=True))
# print( cum_counts.sum(dim=1, keepdims=True).shape)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
# print(estimates.shape)
d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(), label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
d2l.plt.show()


############################
# x = torch.arange(24).reshape(2, 3, 4)
# print(x)
# print(x.sum(dim=0))  # 第0维每个数相加  最外面算整体，
# print(x.sum(dim=0, keepdims=True).shape)  # 第0维每个数相加  最外面算整体，
# print(x.sum(dim=0).shape)  # 第0维每个数相加  最外面算整体，
# print(x.sum(dim=1))
# print(x.sum(dim=1, keepdims=True).shape)
# print(x.sum(dim=1).shape)
# print(x.sum(dim=2))
# print(x.sum(dim=2, keepdims=True).shape)
# print(x.sum(dim=2).shape)
######################
