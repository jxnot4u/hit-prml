import logistic
import os
import pandas as pd
# 获取数据
data_file = os.path.join('.', 'logistic_data2.csv')
data = pd.read_csv(data_file, )
# 增加非线性特征值
y = data['y']  # 取出y列
del data['y']

data['x0**2'] = data['x0'] ** 2
data['x1**2'] = data['x1'] ** 2
data['x0*x1'] = data['x0'] * data['x1']
data['x0**3'] = data['x0'] ** 3
data['x1**4'] = data['x1'] ** 4
data['y'] = y  # 恢复y列，使其保持为最后一列
# 分析
re = logistic.Model(data=data, lr=0.01, epoch=10000, L2=True, alpha=0.01)
theta = re.solve()
print(f'Theta:{theta}')
re.show()
