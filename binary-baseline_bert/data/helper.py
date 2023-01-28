import pandas as pd

train_data = pd.read_csv("train_data.csv", header=0)
# 去除label中值为2的行
train_data = train_data[train_data['label'] != 2]
# 保存
train_data.to_csv("train_data.csv", index=False)

# 读取数据
dev_data = pd.read_csv("dev_data.csv", header=0)
# 去除label中值为2的行
dev_data = dev_data[dev_data['label'] != 2]
# 保存
dev_data.to_csv("dev_data.csv", index=False)

# 读取数据
test_data = pd.read_csv("test_data.csv", header=0)
# 去除label中值为2的行
test_data = test_data[test_data['label'] != 2]
# 保存
test_data.to_csv("test_data.csv", index=False)

