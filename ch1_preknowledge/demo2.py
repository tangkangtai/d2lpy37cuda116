# 数据预处理

import os
import pandas as pd
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
# with open(data_file, 'w') as f:
#     f.write('NumRooms, Ally, Price\n')
#     f.write('NA,Pave,127500\n')
#     f.write('2,NA,106000\n')
#     f.write('4,NA,178100\n')
#     f.write('NA,NA,140000\n')

################################################
data = pd.read_csv(data_file)
print(data)

#“NaN”项代表缺失值。为了处理缺失的数据，典型的⽅法包括插值法和删除法，其中插值法⽤⼀个替
# 代值弥补缺失值，⽽删除法则直接忽略缺失值
inputs, outputs = data.iloc[:,0:2], data.iloc[:,2]
inputs = inputs.fillna(inputs.mean())
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# 转换为张量格式
import torch
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X)
print(y)

