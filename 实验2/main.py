import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

# 超参数
batch_size = 64
lr, num_epochs = 0.05, 35

# tensorboard
writer = SummaryWriter(log_dir='log')

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),            # 调整图像大小
    transforms.ToTensor(),                    # 转换为张量
])

# 加载 Caltech101 数据集
dataset = ImageFolder(root='./caltech-101/101_ObjectCategories', transform=transform)
# 计算数据集大小
dataset_size = len(dataset)
# 计算划分的样本数量
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size
# 随机划分训练集、验证集和测试集
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

net = nn.Sequential(
    # 这里使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.7),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.7),
    # 最后是输出层。由于这里使用caltech-101，所以类别数为101
    nn.Linear(4096, 101))

# 设置设备（使用 GPU 如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)
print('training on', device)
# 定义优化器和损失函数
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()


# 初始化权重
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


net.apply(init_weights)

# 设置训练模式
net.train()
# 迭代训练
total_loss = 0
total_step = len(train_loader)   # 确定步数
for epoch in range(num_epochs):  # 迭代
    for i, (images, labels) in enumerate(train_loader):
        # 将输入数据和标签加载到设备上（如GPU）
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        lo = loss(outputs, labels)
        optimizer.zero_grad()
        lo.backward()
        optimizer.step()
        # 损失函数值求和
        total_loss += lo
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, lo.item()))

        # writer.add_scalar('Train Loss', lo.item(), epoch)
    # 计算平均权重并写入数据
    avr_loss = total_loss / total_step
    writer.add_scalar('Train Loss', avr_loss, epoch)
    total_loss = 0

# 评估模型
net.eval()
# 计算在训练集、验证集和测试集上测试精度
with torch.no_grad():
    correct_train = 0
    total_train = 0
    correct_val = 0
    total_val = 0
    correct_test = 0
    total_test = 0
    # 训练集
    for images_train, labels_train in train_loader:
        images_train = images_train.to(device)
        labels_train = labels_train.to(device)
        outputs_train = net(images_train)
        _, predicted_train = torch.max(outputs_train.data, 1)
        total_train += labels_train.size(0)
        correct_train += (predicted_train == labels_train).sum().item()
        train_acc = correct_train / total_train
    # 验证集
    for images_val, labels_val in val_loader:
        images_val = images_val.to(device)
        labels_val = labels_val.to(device)
        outputs_val = net(images_val)
        _, predicted_val = torch.max(outputs_val.data, 1)
        total_val += labels_val.size(0)
        correct_val += (predicted_val == labels_val).sum().item()
        val_acc = correct_val / total_val
    # 测试集
    for images_test, labels_test in test_loader:
        images_test = images_test.to(device)
        labels_test = labels_test.to(device)
        outputs_test = net(images_test)
        _, predicted_test = torch.max(outputs_test.data, 1)
        total_test += labels_test.size(0)
        correct_test += (predicted_test == labels_test).sum().item()
        test_acc = correct_test / total_test
    # 输出精度结果
    print('训练集测试精度：{:.4f}'.format(train_acc))
    print('验证集测试精度：{:.4f}'.format(val_acc))
    print('测试集测试精度：{:.4f}'.format(test_acc))
