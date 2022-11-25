import numpy as np
import pandas as pd
import torch
import hashlib
import os
import tarfile
import zipfile
import requests
from torch import nn
from d2l import torch as d2l


DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=os.path.join('..','data')):
    """下载⼀个DATA_HUB中的⽂件，返回本地⽂件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True) # os.makedirs() 方法用于递归创建目录
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()  # sha1生成一个160bit的结果，通常用40位的16进制字符串表示
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)

        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True) # 向服务器请求数据，服务器返回的结果是个Response对象
    with open(fname, 'wb') as f:
        f.write(r.content)  #response.content能把Response对象的内容以二进制数据的形式返回，适用于图片、音频、视频的下载
    return fname

DATA_HUB['kaggle_house_train'] = (DATA_URL + 'kaggle_house_pred_train.csv', '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (DATA_URL + 'kaggle_house_pred_test.csv', 'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

# ⽤pandas分别加载包含训练数据和测试数据的两个CSV⽂件。
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

# 训练数据集包括1460个样本，每个样本80个特征和1个标签，⽽测试数据集包含1459个样本，每个样本80个特征。
print(train_data.shape)
print(test_data.shape)

# 看看前四个和最后两个特征，以及相应标签（房价）。
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 第⼀个特征是ID，这有助于模型识别每个训练样本。虽然这很⽅便，但它不携带任何⽤于预测的信息。
# 因此，在将数据提供给模型之前，我们将其从数据集中删除。
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 在开始建模之前，我们需要对数据进⾏预处理。⾸先，我们将所有
# 缺失的值替换为相应特征的平均值。然后，为了将所有特征放在⼀个共同的尺度上，我们通过将特征重新缩
# 放到零均值和单位⽅差来标准化数据
"""
x ← x − µ / σ, µ和σ分别表⽰均值和标准差
"""
# # 若⽆法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)
# # 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)


# 创建两个新的指⽰器特征“MSZoning_RL”和“MSZoning_RM”，其值为0或1。根据独热编码，如果
# “MSZoning”的原始值为“RL”，则：“MSZoning_RL”为1，“MSZoning_RM”为0
# # “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指⽰符特征
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)

# 此转换会将特征的总数量从79个增加到331个。最后，通过values属性，我们可以从pandas格
# 式中提取NumPy格式，并将其转换为张量表⽰⽤于训练
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

# 训练
# 训练⼀个带有损失平⽅的线性模型。显然线性模型很难让我们在竞赛中获胜，但线性模型提供了
# ⼀种健全性检查，以查看数据中是否存在有意义的信息
loss = nn.MSELoss()
in_features = train_features.shape[1]
def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net

# 房价就像股票价格⼀样，我们关⼼的是相对数量，⽽不是绝对数量。因此，我们更关⼼相对误差y−yˆ / y ，⽽不是
# 绝对误差y − yˆ。例如，如果我们在俄亥俄州农村地区估计⼀栋房⼦的价格时，假设我们的预测偏差了10万美
# 元，然⽽那⾥⼀栋典型的房⼦的价值是12.5万美元，那么模型可能做得很糟糕。另⼀⽅⾯，如果我们在加州
# 豪宅区的预测出现同样的10万美元的偏差，（在那⾥，房价中位数超过400万美元）这可能是⼀个不错的预测。
# 解决这个问题的⼀种⽅法是⽤价格预测的对数来衡量差异。事实上，这也是⽐赛中官⽅⽤来评价提交质量的
# 误差指标。即将δ for | log y − log yˆ| ≤ δ 转换为e−δ ≤ˆyy ≤ eδ。
# 这使得预测价格的对数与真实标签价格的对数之间出现以下均⽅根误差：
def log_rmse(net, features, labels):
    # 为了在取对数时进⼀步稳定该值，将⼩于1的值设置为1         # flaot('inf') 无穷
    clipped_preds = torch.clamp(net(features), 1, float('inf')) # torch.clamp(input, min, max)将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()

# 与前⾯的部分不同，我们的训练函数将借助Adam优化器（我们将在后⾯章节更详细地描述它）。
# Adam优化器的主要吸引⼒在于它对初始学习率不那么敏感
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)

    # 这⾥使⽤的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr= learning_rate,
                                 weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()

        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))

    return train_ls, test_ls
# 介绍了K折交叉验证，它有助于模型选择和超参数调整。
# 我们⾸先需要定义⼀个函数，在K折交叉验证过程中返回第i折的数据。
# 具体地说，它选择第i个切⽚作为验证数据，其余部分作为训练数据。
def get_k_fold_data(k, i , X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


# 当我们在K折交叉验证中训练K次后，返回训练和验证误差的平均值。
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum =0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)

        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)

print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, 'f'平均验证log rmse: {float(valid_l):f}')
