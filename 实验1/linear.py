import logistic
import os
import pandas as pd
# 获取数据
data_file = os.path.join('.', 'logistic_data1.csv')
data = pd.read_csv(data_file, )
# 分析
re = logistic.Model(data=data, lr=0.001, epoch=10000)
theta = re.solve()
print(f'Theta:{theta}')
re.show()
