
import math
import torch
from torch import nn
from d2l import torch as d2l
# 我们经常使⽤卷积神经⽹络（CNN）或循环神经⽹络（RNN）对序列进⾏编码。想象⼀下，有
# 了注意⼒机制之后，我们将词元序列输⼊注意⼒汇聚中，以便同⼀组词元同时充当查询、键和值。具体来说，
# 每个查询都会关注所有的键－值对并⽣成⼀个注意⼒输出。由于查询、键和值来⾃同⼀组输⼊，因此被称为
# ⾃注意⼒（self-attention）[Lin et al., 2017b, Vaswani et al., 2017]，

num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
attention.eval()

batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
print(attention(X, X, X, valid_lens).shape)

################位置编码####################
# 为了使⽤序列的顺序信息，我们通过在输⼊表⽰中添加 位置编码（positional encoding）来注⼊绝对的或相对的位置信息
# pi,2j = sin (i / 10000^2j/d)
# pi,2j+1 = cos (i / 10000^2j/d)
# 乍 ⼀ 看， 这 种 基 于 三 ⻆ 函 数 的 设 计 看 起 来 很 奇 怪。 在 解 释 这 个 设 计 之 前， 让 我 们 先 在 下 ⾯
# 的PositionalEncoding类中实现它。
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建⼀个⾜够⻓的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

# 在位置嵌⼊矩阵P中，⾏代表词元在序列中的位置，列代表位置编码的不同维度。在下⾯的例⼦中，我们可以
# 看到位置嵌⼊矩阵的第6列和第7列的频率⾼于第8列和第9列。第6列和第7列之间的偏移量（第8列和第9列相
# 同）是由于正弦函数和余弦函数的交替。
