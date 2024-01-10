import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, recall_score, f1_score
from tensorboardX import SummaryWriter

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyRNN, self).__init__()
        self.hidden_size = hidden_size
        self.Wxh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.Why = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01)
        self.bh = nn.Parameter(torch.zeros(hidden_size))
        self.by = nn.Parameter(torch.zeros(output_size))

    def forward(self, inputs):
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        H = torch.zeros(batch_size, self.hidden_size).to(inputs.device)
        for t in range(seq_len):
            X = inputs[:, t, :]
            H = torch.tanh(torch.mm(X, self.Wxh) + torch.mm(H, self.Whh) + self.bh)
        output = torch.mm(H, self.Why) + self.by
        return output


class TestRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(TestRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 定义RNN层
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # 前向传播
        out, _ = self.rnn(x, h0)
        # 使用全连接层进行分类
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out


class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyGRU, self).__init__()
        self.hidden_size = hidden_size
        self.Wxr = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.Whr = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.Wxz = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.Whz = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.Wxh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.br = nn.Parameter(torch.randn(hidden_size) * 0.01)
        self.bz = nn.Parameter(torch.randn(hidden_size) * 0.01)
        self.bh = nn.Parameter(torch.randn(hidden_size) * 0.01)
        self.Whq = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01)
        self.bq = nn.Parameter(torch.randn(output_size) * 0.01)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        H = torch.zeros(batch_size, self.hidden_size).to(inputs.device)
        for t in range(seq_len):
            X = inputs[:, t, :]
            Z = torch.sigmoid((X @ self.Wxz) + (H @ self.Whz) + self.bz)
            R = torch.sigmoid((X @ self.Wxr) + (H @ self.Whr) + self.br)
            H_tilda = torch.tanh((X @ self.Wxh) + ((R * H) @ self.Whh) + self.bh)
            H = Z * H + (1 - Z) * H_tilda
        output = H @ self.Whq + self.bq
        return output


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.Wxi = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.Whi = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.Wxf = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.Whf = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.Wxo = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.Who = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.Wxc = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.Whc = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bi = nn.Parameter(torch.randn(hidden_size) * 0.01)
        self.bf = nn.Parameter(torch.randn(hidden_size) * 0.01)
        self.bo = nn.Parameter(torch.randn(hidden_size) * 0.01)
        self.bc = nn.Parameter(torch.randn(hidden_size) * 0.01)
        self.Whq = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01)
        self.bq = nn.Parameter(torch.randn(output_size) * 0.01)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        H = torch.zeros(batch_size, self.hidden_size).to(inputs.device)
        C = torch.zeros(batch_size, self.hidden_size).to(inputs.device)
        for t in range(seq_len):
            X = inputs[:, t, :]
            II = torch.sigmoid((X @ self.Wxi) + (H @ self.Whi) + self.bi)
            F = torch.sigmoid((X @ self.Wxf) + (H @ self.Whf) + self.bf)
            OO = torch.sigmoid((X @ self.Wxo) + (H @ self.Who) + self.bo)
            C_tilda = torch.tanh((X @ self.Wxc) + (H @ self.Whc) + self.bc)
            C = F * C + II * C_tilda
            H = OO * torch.tanh(C)
        output = H @ self.Whq + self.bq
        return output


class MyBiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyBiRNN, self).__init__()
        self.hidden_size = hidden_size

        # 正向RNN的参数
        self.Wxh_forward = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.Whh_forward = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bh_forward = nn.Parameter(torch.zeros(hidden_size) * 0.01)

        # 逆向RNN的参数
        self.Wxh_backward = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.Whh_backward = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bh_backward = nn.Parameter(torch.zeros(hidden_size))

        # 输出层的参数
        self.Why = nn.Parameter(torch.randn(hidden_size * 2, output_size) * 0.01)
        self.by = nn.Parameter(torch.zeros(output_size))

    def forward(self, inputs):
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)

        # 正向传播
        H_forward = torch.zeros(batch_size, self.hidden_size).to(inputs.device)
        for t in range(seq_len):
            X = inputs[:, t, :]
            H_forward = torch.tanh(torch.mm(X, self.Wxh_forward) + torch.mm(H_forward, self.Whh_forward) + self.bh_forward)

        # 逆向传播
        H_backward = torch.zeros(batch_size, self.hidden_size).to(inputs.device)
        for t in reversed(range(seq_len)):
            X = inputs[:, t, :]
            H_backward = torch.tanh(torch.mm(X, self.Wxh_backward) + torch.mm(H_backward, self.Whh_backward) + self.bh_backward)

        # 拼接正向和逆向的隐藏状态
        H_concat = torch.cat((H_forward, H_backward), dim=1)

        # 输出层
        output = torch.mm(H_concat, self.Why) + self.by
        return output


def generate_tensor(sentence, maxlen, embedding, word2id):
    tensor = torch.zeros([maxlen, embedding.embedding_dim])
    for index in range(0, maxlen):
        if index >= len(sentence):
            break
        else:
            word = sentence[index]
            if word in word2id:
                vector = embedding.weight[word2id[word]]
                tensor[index] = vector
    return tensor


class MyDataset(Dataset):
    def __init__(self, data, seq_len, embedding, word2id):
        self.x = data['review']
        self.y = data['cat']
        self.seq_len = seq_len
        self.embedding = embedding
        self.word2id = word2id

    def __getitem__(self, index):
        tensor = generate_tensor(self.x[index], self.seq_len, self.embedding, self.word2id)
        return tensor, self.y[index]

    def __len__(self):
        return len(self.x)


def train(model, train_loader, num_epochs, optimizer, criterion, device):
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
            output = model(data)  # 传入数据并前向传播获取输出\
            loss = criterion(output, target)
            loss.backward()
            # 执行梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            # 损失函数值求和
            total_loss += loss
            # 输出
            if (batch_idx + 1) % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, batch_idx + 1,
                                                                         total_step, loss.item()))
            # writer.add_scalar('Train Loss', loss.item(), (epoch + batch_idx / total_step))
        # 计算平均权重并写入数据
        avr_loss = total_loss / total_step
        writer.add_scalar('Train Loss', avr_loss, epoch)
        total_loss = 0


def test(model, test_loader, device):
    # 测试模型
    model.eval()
    labels_test = []
    predicted_test = []
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            # GPU上的tensor张量无法转为numpy格式
            labels_test.extend(labels.cpu().numpy())
            predicted_test.extend(predicted.cpu().numpy())
    # 计算准确率
    accuracy = accuracy_score(labels_test, predicted_test)
    # 计算召回率
    recall = recall_score(labels_test, predicted_test, average='macro')
    # 计算F1值
    f1 = f1_score(labels_test, predicted_test, average='macro')
    return accuracy, recall, f1
