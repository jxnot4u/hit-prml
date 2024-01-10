import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 损失函数
def compute_cost(X, y, theta, L2, alpha):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = (1 / m) * np.sum(-y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon))
    if L2:
        cost += (alpha / (2*m)) * np.sum(theta**2)
    return cost


# logistic回归模型，输入的data里面的特征值名符合运算式
class Model:
    def __init__(self, data, lr, epoch, L2=False, alpha=0):
        self.lr = lr
        self.epoch = epoch
        self.data = data
        self.X = data.iloc[:, :-1].values  # 提取特征列（除了最后一列）
        self.y = data.iloc[:, -1].values  # 提取目标变量列（最后一列）
        self.X, self.mean, self.std = self.processData()
        self.theta = np.zeros((len(self.X[1]))).T
        self.L2 = L2
        self.alpha = alpha

    # 处理数据：特征规范化、增广
    def processData(self):
        mean = np.mean(self.X, axis=0)  # 沿着第0维度（行）计算平均值
        std = np.std(self.X, axis=0)
        self.X = (self.X - mean) / std
        X = np.insert(self.X, 0, 1, axis=1)
        return X, mean, std

    # 梯度下降
    def gradient_descent(self):
        m = len(self.y)
        X = self.X
        y = self.y
        lr = self.lr
        alpha = self.alpha
        cost_history = []
        for i in range(self.epoch):
            h = sigmoid(X @ self.theta)
            if self.L2:
                self.theta -= (lr / m) * ((X.T @ (h - y)) - (alpha * self.theta))
            else:
                self.theta -= (lr / m) * (X.T @ (h - y))
            cost = compute_cost(X, y, self.theta, self.L2, self.alpha)
            cost_history.append(cost)
        # 输出最终损失函数值
        print(f'Final Loss:{cost_history[-1]}')
        # 显示损失函数图像
        plt.plot(range(self.epoch), cost_history)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

    # 恢复因为特征规范化影响的theta
    def retheta(self):
        self.theta[1:] = self.theta[1:] / self.std
        self.theta[0] -= np.dot(self.mean, self.theta[1:])

    # 求解优化值
    def solve(self):
        self.gradient_descent()
        self.retheta()
        return self.theta

    # 显示分类界面图像
    def show(self):
        data = self.data
        theta = self.theta
        data_0 = data[data['y'] == 0]
        data_1 = data[data['y'] == 1]
        # 数据图
        plt.scatter(data_0['x0'], data_0['x1'], c='#1f77b4')  # blue
        plt.scatter(data_1['x0'], data_1['x1'], c='#ff7f0e')  # orange
        plt.legend(['y = 0', 'y = 1'])
        # 获取图像显示范围
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        # 画出模型方程值为0时等高线——分界线
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x, y)
        Z = theta[0] * np.ones_like(X)
        # 由数据特征值名称恢复模型方程
        variables = {'x0': X, 'x1': Y}
        headers = data.columns.tolist()
        for i in range(len(headers)-1):
            Z += theta[i+1] * eval(headers[i], variables)
        plt.contour(X, Y, Z, levels=[0])
        plt.show()

    def predict(self):
        ...


# if __name__ == '__main__':
#     data_file = os.path.join('.', 'logistic_data1.csv')
#     data = pd.read_csv(data_file, )
#
#     re = Model(data=data, lr=0.001, epoch=100000)
#     theta = re.solve()
#     print(theta)
#
#     data_0 = data[data['y'] == 0]
#     data_1 = data[data['y'] == 1]
#
#     plt.scatter(data_0['x0'], data_0['x1'], c='#1f77b4')  # blue
#     plt.scatter(data_1['x0'], data_1['x1'], c='#ff7f0e')  # orange
#     plt.legend(['y = 0', 'y = 1'])
#
#     x_min = data['x0'].min()
#     x_max = data['x0'].max()
#     y_min = data['x1'].min()
#     y_max = data['x1'].max()
#
#     x = np.linspace(x_min, x_max, 10)
#     y = np.linspace(y_min, y_max, 100)
#     # plt.plot(a, b)
#     X, Y = np.meshgrid(x, y)
#     Z = theta[0] + theta[1] * X + theta[2] * Y
#     plt.contour(X, Y, Z, levels=[0])
#     # 设置 x 轴和 y 轴的范围
#     plt.xlim(x_min - 5, x_max + 5)  # 设置 x 轴的范围，x_min和x_max是最小值和最大值
#     plt.ylim(y_min - 5, y_max + 5)  # 设置 y 轴的范围，y_min和y_max是最小值和最大值
#     plt.show()
