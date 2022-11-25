
import torch
from d2l import torch as d2l
# Nadaraya-Waston核回归（kernel regression）正是具有 注意⼒机制（attention mechanism）的机器学习的简单演⽰

# “是否包含⾃主性提⽰”将注意⼒机制与全连接层或汇聚层区别开来。
# 在注意⼒机制的背景下，我们将 ⾃主性提⽰ 称为 查询（query）。
# 给定任何 查询，注意⼒机制通过注意⼒汇聚（attention pooling）将选择引导⾄感官输⼊（sensory inputs，例如中间特征表⽰）。
# 在注意⼒机制中，这些感官输⼊被称为 值（value）。
# 更通俗的解释，每个值都与⼀个键（key）配对，这可以想象为感官输⼊的⾮⾃主提⽰。
# 如图10.1.3所⽰，我们可以设计注意⼒汇聚，以便给定的查询（⾃主性提⽰）可以与键（⾮⾃主性提⽰）进⾏匹配，这将引导得
# 出最匹配的值（感官输⼊）

# 注意⼒机制通过注意⼒汇聚将查询（⾃主性提⽰）和键（⾮⾃主性提⽰）结合在⼀起，实现对值（感官输⼊）的选择倾向

# 注意⼒的可视化

attention_weights = torch.eye(10).reshape((1, 1, 10, 10))  # 生成对角矩阵
