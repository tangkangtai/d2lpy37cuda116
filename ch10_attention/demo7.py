import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

# transformer
# Transformer作为编码器－解码器架构的⼀个实例，transformer是由编码器和解码器组成的
# transformer的编码器和解码器是基于⾃注意⼒的模块叠加⽽成的
# 源（输⼊）序列和⽬标（输出）序列的嵌⼊（embedding）表⽰将加上位置编码（positional encoding），再分别输⼊到编码器和解码器中。
############################################################################
# Transformer解码器也是由多个相同的层叠加⽽成的，并且层中使⽤了残差连接和层规范化。除了编码器中
# 描述的两个⼦层之外，解码器还在这两个⼦层之间插⼊了第三个⼦层，称为编码器－解码器注意⼒（encoder-decoder attention）层。
# 在编码器－解码器注意⼒中，查询来⾃前⼀个解码器层的输出，⽽键和值来⾃整个编码器的输出。
# 在解码器⾃注意⼒中，查询、键和值都来⾃上⼀个解码器层的输出。
# 但是，解码器中的每个位置只能考虑该位置之前的所有位置。
# 这种掩蔽（masked）注意⼒保留了⾃回归（auto-regressive）属性，确
# 保预测仅依赖于已⽣成的输出词元。

############基于位置的前馈⽹络####################################
# 基于位置的前馈⽹络对序列中的所有位置的表⽰进⾏变换时使⽤的是同⼀个多层感知机（MLP），这就是称前馈⽹络是基于位置的（positionwise）的原因。
# 在下⾯的实现中，输⼊X的形状（批量⼤⼩，时间步数或序列⻓度，隐单元数或特征维度）将被⼀个两层的感知机转换成形状为（批量⼤⼩，时间步数，ffn_num_outputs）
# 的输出张量。

class PositionWiseFFN(nn.Module):
    """基于位置的前馈⽹络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

# 改变张量的最⾥层维度的尺⼨，会改变成基于位置的前馈⽹络的输出尺⼨。
# 因为⽤同⼀个多层感知机对所有位置上的输⼊进⾏变换，所以当所有这些位置的输⼊相同时，它们的输出也是相同的。
ffn = PositionWiseFFN(4, 4, 8)
ffn.eval()
# print(ffn(torch.ones((2, 3, 4))).shape)

# “加法和规范化（add&norm）”组件。
# 正如在本节开头所述，这是由残差连接和紧随其后的层规范化组成的。两者都是构建有效的深度架构的关键
# 我们解释了在⼀个⼩批量的样本内基于批量规范化对数据进⾏重新中⼼化和重新缩放的调整。

# 层规范化和批量规范化的⽬标相同，但层规范化是基于特征维度进⾏规范化
ln = nn.LayerNorm(2)
bn = nn.BatchNorm1d(2)
X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
# print('layer norm:', ln(X), '\nbatch norm:', bn(X))

class AddNorm(nn.Module):
    """残差连接后进⾏层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

add_norm = AddNorm([3, 4], 0.5)
add_norm.eval()
print(add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape)

# ##########################编码器#################################
# 有了组成transformer编码器的基础组件，现在可以先实现编码器中的⼀个层。下⾯的EncoderBlock类包
# 含两个⼦层：多头⾃注意⼒和基于位置的前馈⽹络，这两个⼦层都使⽤了残差连接和紧随的层规范化

class EncoderBlock(nn.Module):
    """transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
# 正如我们所看到的，transformer编码器中的任何层都不会改变其输⼊的形状
X = torch.ones((2, 100, 24))
valid_lens = torch.tensor([3, 2])
encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
encoder_blk.eval()
print(encoder_blk(X, valid_lens).shape)
# 在实现下⾯的transformer编码器的代码中，我们堆叠了num_layers个EncoderBlock类的实例。
# 由于我们使⽤的是值范围在−1和1之间的固定位置编码，因此通过学习得到的输⼊的嵌⼊表⽰的值需要先乘以嵌⼊
# 维度的平⽅根进⾏重新缩放，然后再与位置编码相加。
class TransformerEncoder(d2l.Encoder):
    """transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens,num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block"+str(i), EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                             norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌⼊值乘以嵌⼊维度的平⽅根进⾏缩放，
        # 然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            x = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X