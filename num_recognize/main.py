import requests
import pickle
import gzip
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from pathlib import Path
from matplotlib import pyplot

##下载Mnist数据集
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
PATH.mkdir(parents=True, exist_ok=True)
URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"
if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

##读取数据
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

##看看数据长什么样子,展示这个灰度图
pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
##print(x_train.shape) 50000 * 784(28*28*1)


##读取的数据是数组格式，转化成tensor格式
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
#x_train, x_train.shape, y_train.min(), y_train.max()
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())

##定义损失函数
loss_func = F.cross_entropy

##测试这个函数好不好用，xb是输入数据，mm方法是矩阵乘法，就是定义了一个XW+b的一个函数
def model(xb):
    return xb.mm(weights) + bias

bs = 64  #  一次训练样本的个数
xb = x_train[0:bs]  # 从训练集中取前bs（64）张图
yb = y_train[0:bs]  # 从训练集中取前bs（64）个标签
weights = torch.randn([784, 10], dtype=torch.float, requires_grad=True)  #  随机初始化设置W为[784*10]的矩阵，10分类
bias = torch.zeros(10, requires_grad=True)  #  初始化偏置bias

print(loss_func(model(xb), yb))


class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)  #
        self.hidden2 = nn.Linear(128, 256)
        self.out = nn.Linear(256, 10)
        #self.dropout = nn.Linear(0.5)  #  dropout防止过拟合,0.5表示随机丢1半

# x就是一会会输进来的数据，比如[64 * 784]的一个矩阵，64张图，每张图784个像素点
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x

# 看看刚刚定义的这个模型
net = Mnist_NN()
print(net)
# 返回这个类的名字，和实际的参数值
for name, parameter in net.named_parameters():
    # 打印名字(哪层hidden1？)；打印权重参数值；打印权重参数大小
    print(name, parameter, parameter.size())


# 定义数据集
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


# steps:迭代多少轮; model:定义的模型; loss_func:损失函数; opt:优化器（梯度下降）; train_dl, valid_dl:打包器
def fit(steps, model, loss_func, opt, train_dl, valid_dl):
    for step in range(steps):
        model.train()  # 训练模式，更新w和b
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()  # 验证模式
        with torch.no_grad():
            # 返回当前损失和数字
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)  # 平均损失
        print('当前step:' + str(step), '验证集损失：' + str(val_loss))


def get_model():
    model = Mnist_NN()
    return model, optim.Adam(model.parameters(), lr=0.001)  # lr学习率

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()  # 反向传播，算每一层(hidden)的权重参数，有梯度了
        opt.step()  # 更新，沿着梯度反向，走学习率这个大小去更新
        opt.zero_grad()  # 清空之前更新的梯度的累加
    return loss.item(), len(xb)  # 返回损失，和 训练样本有多少个

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(20, model, loss_func, opt, train_dl, valid_dl)  # 20是epoch，就是几批bs,也可以说迭代几次

correct = 0
total = 0
for xb, yb in valid_dl:
    outputs = model(xb)
    _,predicted = torch.max(outputs.data,1)
    total += yb.size(0)
    correct += (predicted == yb).sum().item()

print('Accuracy of the network on the 10000 test images: %d%%' % (100 * correct / total))

# 读取自己的图片来识别
def convert_to_float32(value):
    return np.float32(value)

from PIL import Image
img6 = Image.open("xu8.png")
img6 = img6.convert("L")
img6 = np.array(img6) # 转化为numpy
img6 = convert_to_float32(img6)
img_new = torch.from_numpy(img6) # 转化为tensor
img_new = img_new.reshape(1, 784) # 维度变化（降维）

# 用我的模型预测输出
img_new = model(img_new)
_,predicted = torch.max(img_new.data,1)
print(predicted)
print("你输入的是：", predicted.item())