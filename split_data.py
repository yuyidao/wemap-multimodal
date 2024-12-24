import pandas as pd
from sklearn.model_selection import train_test_split

# 读取原始 CSV 文件
data = pd.read_csv('data.csv')

# 使用 train_test_split 进行数据划分
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])

# 将划分好的数据保存为新的 CSV 文件
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

# 打印划分后的数据集大小，检查是否按比例划分
print(f"Train data size: {len(train_data)}")
print(f"Test data size: {len(test_data)}")