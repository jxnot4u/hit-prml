import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from mymodels import MyLSTM
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

batch_size = 64
num_epochs = 50
lr = 1
hidden_size = 512


# 自定义数据集类
class CliDataset(Dataset):
    def __init__(self, data):
        t = data['t'].tolist()
        self.x = []
        self.y = []
        lim = len(t)
        i = 0
        xnum = 144 * 5
        ynum = 144 * 7
        while i+ynum < lim:
            if i + xnum <= lim:
                self.x.append(torch.tensor(t[i:i + xnum]).unsqueeze(-1))
            if i + ynum <= lim:
                self.y.append(torch.tensor(t[i + xnum:i + ynum]).unsqueeze(-1))
            i += ynum

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


# 获取数据
data = pd.read_csv('jena_climate_2009_2016.csv', usecols=[0, 2])
data.rename(columns={'Date Time': 'dt', 'T (degC)': 't'}, inplace=True)
train_data = data[data['dt'].str.contains('2009|2010|2011|2012|2013|2014')]
test_data = data[data['dt'].str.contains('2015|2016')]
test_data.reset_index(drop=True, inplace=True)

train_set = CliDataset(train_data)
test_set = CliDataset(test_data)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# 设置设备（使用 GPU 如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型
model = MyLSTM(input_size=1, hidden_size=hidden_size, output_size=2 * 144)
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)
print(f'开始训练，设备：{device}')

# 将模型设置为训练模式
model.train()
# tensorboard
writer = SummaryWriter(log_dir='log')
total_loss = 0
# 确定步数
total_step = len(train_loader)
max_grad_norm = 0.1  # 最大裁剪范数
# 迭代训练
for epoch in range(num_epochs):
    # 执行一次训练迭代
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()  # 清除所有优化的梯度
        output = model(data)  # 传入数据并前向传播获取输出
        target = target.squeeze()
        loss = criterion(output, target)
        loss.backward()
        # 执行梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        # 损失函数值求和
        total_loss += loss
        # 输出
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, batch_idx + 1,
                                                                 total_step, loss.item()))
        # writer.add_scalar('Train Loss', loss.item(), (epoch + batch_idx / total_step))
    # 计算平均权重并写入数据
    avr_loss = total_loss / total_step
    writer.add_scalar('Train Loss', avr_loss, epoch)
    total_loss = 0
#
# # 保存模型
# torch.save(model.state_dict(), 'cli.pth')

# # 加载模型参数
# model.load_state_dict(torch.load('cli.pth'))

# 测试模型
model.eval()
with torch.no_grad():
    i = 1
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        target = target.squeeze()
        outputs = model(data)
        # 计算平均误差和中位误差
        mae = torch.mean(torch.abs(outputs - target))
        mad = torch.median(torch.abs(outputs - target))
        print('Batch{}：平均误差{:.4f}，中位误差{:.4f}'.format(i, mae.item(), mad.item()))
        i += 1

predict_loader = DataLoader(test_set, batch_size=1, shuffle=True)
true = []
predict = []
with torch.no_grad():
    i = 1
    for data, target in predict_loader:
        if i > 4:
            break
        data = data.to(device)
        target = target.to(device)
        outputs = model(data)

        data = data.squeeze()
        target = target.squeeze()
        outputs = outputs.squeeze()

        target = torch.cat((data, target), dim=0)
        outputs = torch.cat((data, outputs), dim=0)

        # GPU上的tensor张量无法转为numpy格式
        true.append(target.cpu().numpy())
        predict.append(outputs.cpu().numpy())
        i += 1

fig, axes = plt.subplots(nrows=2, ncols=2)
# 数据设置
x = np.linspace(1, 8, 1008)

# 在每个子图中绘制折线图
axes[0, 0].plot(x, true[0], 'b', label='True')
axes[0, 0].plot(x, predict[0], 'r', label='Predict')
axes[0, 0].legend()  # 显示图例

axes[0, 1].plot(x, true[1], 'b', label='True')
axes[0, 1].plot(x, predict[1], 'r', label='Predict')
axes[0, 1].legend()

axes[1, 0].plot(x, true[2], 'b', label='True')
axes[1, 0].plot(x, predict[2], 'r', label='Predict')
axes[1, 0].legend()

axes[1, 1].plot(x, true[3], 'b', label='True')
axes[1, 1].plot(x, predict[3], 'r', label='Predict')
axes[1, 1].legend()

# 调整子图之间的间距
fig.tight_layout()

# 显示图形对象
plt.show()
