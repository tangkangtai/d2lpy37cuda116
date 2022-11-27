
import math
import torch
from torch import nn
from d2l import torch as d2l
# 多头注意⼒
# 在实践中，当给定相同的查询、键和值的集合时，我们希望模型可以基于相同的注意⼒机制学习到不同的⾏
# 为，然后将不同的⾏为作为知识组合起来，捕获序列内各种范围的依赖关系（例如，短距离依赖和⻓距离依
# 赖关系）

############## 多头注意⼒#######################
# 与其只使⽤单独⼀个注意⼒汇聚，我们可以⽤独⽴学习得到的h组不同的 线性投影（linear projections）来变换查询、键和值。
# 然后，这h组变换后的查询、键和值将并⾏地送到注意⼒汇聚中。
# 最后，将这h个注意⼒汇聚的输出拼接在⼀起，并且通过另⼀个可以学习的线性投影进⾏变换，以产⽣最终输出。
##########################################################
# 于h个注意⼒汇聚输出，每⼀个注意⼒汇聚都被称作⼀个头（head）。
# 图10.5.1 展⽰了使⽤全连接层来实现可学习的线性变换的多头注意⼒。

# 我们选择缩放点积注意⼒作为每⼀个注意⼒头。为了避免计算代价和参数代价的⼤幅增⻓，
# 我们设定pq = pk = pv = po/h。
# 值得注意的是，如果我们将查询、键和值的线性变换的输出数量设置为
# pqh = pkh = pvh = po，则可以并⾏计算h个头。在下⾯的实现中，po是通过参数num_hiddens指定的。
class MultiHeadAttention(nn.Module):
    """多头注意⼒"""
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状: (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens的形状: (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values的形状:(batch_size*num_heads，查询或者“键－值”对的个数， num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            # 在轴0，将第⼀项（标量或者⽮量）复制num_heads次，
            # 然后如此复制第⼆项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)
        # output的形状:(batch_size*num_heads，查询的个数，num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

# 为了能够使多个头并⾏计算，上⾯的MultiHeadAttention类将使⽤下⾯定义的两个转置函数。具体来说，
# transpose_output函数反转了transpose_qkv函数的操作
def transpose_qkv(X, num_heads):
    """为了多注意⼒头的并⾏计算⽽变换形状"""
    # 输⼊X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # # 最终输出的形状:(batch_size*num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    return X.reshape(X.shape[0], X.shape[1], -1)


num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
attention.eval()
# print(attention)
batch_size, num_queries = 2, 4
num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
print(X.shape)
print(Y.shape)
print(attention(X, Y, Y, valid_lens).shape)

# 多头注意⼒融合了来⾃于多个注意⼒汇聚的不同知识，这些知识的不同来源于相同的查询、键和值的
# 不同的⼦空间表⽰。


