---
date: '2025-07-08T16:43:43+08:00'
draft: false
title: 'Pytorch入门学习'
tags: ["Pytorch", "深度学习"]
catagoties: ["学习"]
---

本文只会写一些笔记，不包含完整的内容，可以参考[菜鸟教程](https://www.runoob.com/pytorch/pytorch-tutorial.html)

# 关于pytorch中的神经网络

PyTorch 提供了强大的工具来构建和训练神经网络。
神经网络在 PyTorch 中是通过 ```torch.nn``` 模块来实现的。
```torch.nn``` 模块提供了各种网络层（如全连接层、卷积层等）、损失函数和优化器，让神经网络的构建和训练变得更加方便。

在 PyTorch 中，构建神经网络通常需要继承 nn.Module 类。
nn.Module 是所有神经网络模块的基类，你需要定义以下两个部分：
```__init__()```：定义网络层。
```forward()```：定义数据的前向传播过程。

## 简单的全连接神经网络（Fully Connected Network）：
```python
import torch
import torch.nn as nn

# 定义一个简单的神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        # 调用父类的构造函数
        super(SimpleNN, self).__init__()
        # 定义一个输入层到隐藏层的全连接层，fc1的意思是"fully connected layer 1"
        self.fc1 = nn.Linear(2, 2)  # 输入 2 个特征，输出 2 个特征
        # 定义一个隐藏层到输出层的全连接层
        self.fc2 = nn.Linear(2, 1)  # 输入 2 个特征，输出 1 个预测值
    
    def forward(self, x):
        # 前向传播过程
        x = torch.relu(self.fc1(x))  # 使用 ReLU 激活函数
        x = self.fc2(x)  # 输出层
        return x

# 创建模型实例
model = SimpleNN()

# 打印模型
print(model)
```

PyTorch 提供了许多常见的神经网络层，以下是几个常见的：
```nn.Linear(in_features, out_features)```：全连接层，输入 in_features 个特征，输出 out_features 个特征。
```nn.Conv2d(in_channels, out_channels, kernel_size)```：2D 卷积层，用于图像处理。
```nn.MaxPool2d(kernel_size)```：2D 最大池化层，用于降维。
```nn.ReLU()```：ReLU 激活函数，常用于隐藏层。
```nn.Softmax(dim)```：Softmax 激活函数，通常用于输出层，适用于多类分类问题。

## 激活函数（Activation Function）
激活函数决定了神经元是否应该被激活。它们是非线性函数，使得神经网络能够学习和执行更复杂的任务。常见的激活函数包括：
Sigmoid：用于二分类问题，输出值在 0 和 1 之间。
Tanh：输出值在 -1 和 1 之间，常用于输出层之前。
ReLU（Rectified Linear Unit）：目前最流行的激活函数之一，定义为 f(x) = max(0, x)，有助于解决梯度消失问题。
Softmax：常用于多分类问题的输出层，将输出转换为概率分布。

```python
import torch.nn.functional as F

# ReLU 激活
output = F.relu(input_tensor)

# Sigmoid 激活
output = torch.sigmoid(input_tensor)

# Tanh 激活
output = torch.tanh(input_tensor)
```

## 损失函数（Loss Function）
损失函数用于衡量模型的预测值与真实值之间的差异。
常见的损失函数包括：
均方误差（MSELoss）：回归问题常用，计算输出与目标值的平方差。
交叉熵损失（CrossEntropyLoss）：分类问题常用，计算输出和真实标签之间的交叉熵。
BCEWithLogitsLoss：二分类问题，结合了 Sigmoid 激活和二元交叉熵损失。
```python
# 均方误差损失
criterion = nn.MSELoss()

# 交叉熵损失
criterion = nn.CrossEntropyLoss()

# 二分类交叉熵损失
criterion = nn.BCEWithLogitsLoss()
```

## 优化器（Optimizer）
优化器负责在训练过程中更新网络的权重和偏置。
常见的优化器包括：
SGD（随机梯度下降）
Adam（自适应矩估计）
RMSprop（均方根传播）
```python
import torch.optim as optim

# 使用 SGD 优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 使用 Adam 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

# 训练过程（Training Process）
训练神经网络涉及以下步骤：
1. 准备数据：通过 DataLoader 加载数据。
2. 定义损失函数和优化器。
3. 前向传播：计算模型的输出。
4. 计算损失：与目标进行比较，得到损失值。
5. 反向传播：通过 loss.backward() 计算梯度。
6. 更新参数：通过 optimizer.step() 更新模型的参数。
7. 重复上述步骤，直到达到预定的训练轮数。
```python
# 假设已经定义好了模型、损失函数和优化器

# 训练数据示例
X = torch.randn(10, 2)  # 10 个样本，每个样本有 2 个特征
Y = torch.randn(10, 1)  # 10 个目标标签

# 训练过程
for epoch in range(100):  # 训练 100 轮
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()  # 清除梯度
    output = model(X)  # 前向传播，这种写法就是调用了model的forward函数，forward函数是在定义模型时定义的
    loss = criterion(output, Y)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新权重
    
    if (epoch + 1) % 10 == 0:  # 每 10 轮输出一次损失
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')
```

## 测试与评估
训练完成后，需要对模型进行测试和评估。
常见的步骤包括：
计算测试集的损失：测试模型在未见过的数据上的表现。
计算准确率（Accuracy）：对于分类问题，计算正确预测的比例。
```python
# 假设你有测试集 X_test 和 Y_test
model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 在评估过程中禁用梯度计算
    output = model(X_test)
    loss = criterion(output, Y_test)
    print(f'Test Loss: {loss.item():.4f}')
```

# Pytorch第一个神经网络测试
```python
# 导入PyTorch库
import torch
import torch.nn as nn

# 定义输入层大小、隐藏层大小、输出层大小和批量大小
n_in, n_h, n_out, batch_size = 10, 5, 1, 10

# 创建虚拟输入数据和目标数据
x = torch.randn(batch_size, n_in)  # 随机生成输入数据
y = torch.tensor([[1.0], [0.0], [0.0], 
                 [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])  # 目标输出数据

# 创建顺序模型，包含线性层、ReLU激活函数和Sigmoid激活函数
model = nn.Sequential(
   nn.Linear(n_in, n_h),  # 输入层到隐藏层的线性变换
   nn.ReLU(),            # 隐藏层的ReLU激活函数
   nn.Linear(n_h, n_out),  # 隐藏层到输出层的线性变换
   nn.Sigmoid()           # 输出层的Sigmoid激活函数
)

# 定义均方误差损失函数和随机梯度下降优化器
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 学习率为0.01

# 执行梯度下降算法进行模型训练
for epoch in range(50):  # 迭代50次
   y_pred = model(x)  # 前向传播，计算预测值
   loss = criterion(y_pred, y)  # 计算损失
   print('epoch: ', epoch, 'loss: ', loss.item())  # 打印损失值

   optimizer.zero_grad()  # 清零梯度
   loss.backward()  # 反向传播，计算梯度
   optimizer.step()  # 更新模型参数
```

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 生成一些随机数据
n_samples = 100
data = torch.randn(n_samples, 2)  # 生成 100 个二维数据点
labels = (data[:, 0]**2 + data[:, 1]**2 < 1).float().unsqueeze(1)  # 点在圆内为1，圆外为0

# 可视化数据
plt.scatter(data[:, 0], data[:, 1], c=labels.squeeze(), cmap='coolwarm')
plt.title("Generated Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 定义前馈神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 定义神经网络的层
        self.fc1 = nn.Linear(2, 4)  # 输入层有 2 个特征，隐藏层有 4 个神经元
        self.fc2 = nn.Linear(4, 1)  # 隐藏层输出到 1 个神经元（用于二分类）
        self.sigmoid = nn.Sigmoid()  # 二分类激活函数

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用 ReLU 激活函数
        x = self.sigmoid(self.fc2(x))  # 输出层使用 Sigmoid 激活函数
        return x

# 实例化模型
model = SimpleNN()

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二元交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.1)  # 使用随机梯度下降优化器

# 训练
epochs = 100
for epoch in range(epochs):
    # 前向传播
    outputs = model(data)
    loss = criterion(outputs, labels)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每 10 轮打印一次损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 可视化决策边界
def plot_decision_boundary(model, data):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, 0.1), torch.arange(y_min, y_max, 0.1), indexing='ij')
    grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)
    predictions = model(grid).detach().numpy().reshape(xx.shape)
    plt.contourf(xx, yy, predictions, levels=[0, 0.5, 1], cmap='coolwarm', alpha=0.7)
    plt.scatter(data[:, 0], data[:, 1], c=labels.squeeze(), cmap='coolwarm', edgecolors='k')
    plt.title("Decision Boundary")
    plt.show()

plot_decision_boundary(model, data)
```

# Pytorch数据处理与加载

PyTorch 数据处理与加载的介绍：
- 自定义 Dataset：通过继承 torch.utils.data.Dataset 来加载自己的数据集。
- DataLoader：DataLoader 按批次加载数据，支持多线程加载并进行数据打乱。
- 数据预处理与增强：使用 torchvision.transforms 进行常见的图像预处理和增强操作，提高模型的泛化能力。
- 加载标准数据集：torchvision.datasets 提供了许多常见的数据集，简化了数据加载过程。
- 多个数据源：通过组合多个 Dataset 实例来处理来自不同来源的数据。

## 自定义 Dataset
torch.utils.data.Dataset 是一个抽象类，允许你从自己的数据源中创建数据集。
我们需要继承该类并实现以下两个方法：
- __len__(self)：返回数据集中的样本数量。
- __getitem__(self, idx)：通过索引返回一个样本。
假设我们有一个简单的 CSV 文件或一些列表数据，我们可以通过继承 Dataset 类来创建自己的数据集
```python
import torch
from torch.utils.data import Dataset

# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, X_data, Y_data):
        """
        初始化数据集，X_data 和 Y_data 是两个列表或数组
        X_data: 输入特征
        Y_data: 目标标签
        """
        self.X_data = X_data
        self.Y_data = Y_data

    def __len__(self):
        """返回数据集的大小"""
        return len(self.X_data)

    def __getitem__(self, idx):
        """返回指定索引的数据"""
        x = torch.tensor(self.X_data[idx], dtype=torch.float32)  # 转换为 Tensor
        y = torch.tensor(self.Y_data[idx], dtype=torch.float32)
        return x, y

# 示例数据
X_data = [[1, 2], [3, 4], [5, 6], [7, 8]]  # 输入特征
Y_data = [1, 0, 1, 0]  # 目标标签

# 创建数据集实例
dataset = MyDataset(X_data, Y_data)
```

## DataLoader
DataLoader 是 PyTorch 提供的一个重要工具，用于从 Dataset 中按批次（batch）加载数据。
DataLoader 允许我们批量读取数据并进行多线程加载，从而提高训练效率。
```python
from torch.utils.data import DataLoader

# 创建 DataLoader 实例，batch_size 设置每次加载的样本数量
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 打印加载的数据
for epoch in range(1):
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        print(f'Batch {batch_idx + 1}:')
        print(f'Inputs: {inputs}')
        print(f'Labels: {labels}')
```

- batch_size: 每次加载的样本数量。
- shuffle: 是否对数据进行洗牌，通常训练时需要将数据打乱。
- drop_last: 如果数据集中的样本数不能被 batch_size 整除，设置为 True 时，丢弃最后一个不完整的 batch。

## 数据预处理与增强
PyTorch 提供了 torchvision.transforms 模块来进行常见的图像预处理和增强操作，如旋转、裁剪、归一化等。
```python
import torchvision.transforms as transforms
from PIL import Image

# 定义数据预处理的流水线
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 将图像调整为 128x128
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 加载图像
image = Image.open('image.jpg')

# 应用预处理
image_tensor = transform(image)
print(image_tensor.shape)  # 输出张量的形状
```

- transforms.Compose()：将多个变换操作组合在一起。
- transforms.Resize()：调整图像大小。
- transforms.ToTensor()：将图像转换为 PyTorch 张量，值会被归一化到 [0, 1] 范围。
- transforms.Normalize()：标准化图像数据，通常使用预训练模型时需要进行标准化处理。

数据增强技术通过对训练数据进行随机变换，增加数据的多样性，帮助模型更好地泛化。例如，随机翻转、旋转、裁剪等。
```python
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(30),  # 随机旋转 30 度
    transforms.RandomResizedCrop(128),  # 随机裁剪并调整为 128x128
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## 加载标准数据集
对于图像数据集，torchvision.datasets 提供了许多常见数据集（如 CIFAR-10、ImageNet、MNIST 等）以及用于加载图像数据的工具。
```python
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 定义预处理操作
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 对灰度图像进行标准化
])

# 下载并加载 MNIST 数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 迭代训练数据
for inputs, labels in train_loader:
    print(inputs.shape)  # 每个批次的输入数据形状
    print(labels.shape)  # 每个批次的标签形状
```

- datasets.MNIST() 会自动下载 MNIST 数据集并加载。
- transform 参数允许我们对数据进行预处理。
- train=True 和 train=False 分别表示训练集和测试集。

## 多个数据源
```python
from torch.utils.data import ConcatDataset

# 假设 dataset1 和 dataset2 是两个 Dataset 对象
combined_dataset = ConcatDataset([dataset1, dataset2])
combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)
```

# 线性回归

## 数据准备

我们首先准备一些假数据，用于训练我们的线性回归模型。这里，我们可以生成一个简单的线性关系的数据集，其中每个样本有两个特征 x1，x2。
```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# 随机种子，确保每次运行结果一致
torch.manual_seed(42)

# 生成训练数据
X = torch.randn(100, 2)  # 100 个样本，每个样本 2 个特征
true_w = torch.tensor([2.0, 3.0])  # 假设真实权重
true_b = 4.0  # 偏置项
Y = X @ true_w + true_b + torch.randn(100) * 0.1  # 加入一些噪声

# 打印部分数据
print(X[:5])
print(Y[:5])
```

这段代码创建了一个带有噪声的线性数据集，输入 X 为 100x2 的矩阵，每个样本有两个特征，输出 Y 由真实的权重和偏置生成，并加上了一些随机噪声。

## 定义线性回归模型

我们可以通过继承 nn.Module 来定义一个简单的线性回归模型。在 PyTorch 中，线性回归的核心是 nn.Linear() 层，它会自动处理权重和偏置的初始化。
```python
import torch.nn as nn

# 定义线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # 定义一个线性层，输入为2个特征，输出为1个预测值
        self.linear = nn.Linear(2, 1)  # 输入维度2，输出维度1
    
    def forward(self, x):
        return self.linear(x)  # 前向传播，返回预测结果

# 创建模型实例
model = LinearRegressionModel()
```

## 定义损失函数与优化器

线性回归的常见损失函数是 均方误差损失（MSELoss），用于衡量预测值与真实值之间的差异。PyTorch 中提供了现成的 MSELoss 函数。
我们将使用 SGD（随机梯度下降） 或 Adam 优化器来最小化损失函数。
```python
# 损失函数（均方误差）
criterion = nn.MSELoss()

# 优化器（使用 SGD 或 Adam）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 学习率设置为0.01
```

## 训练模型

在训练过程中，我们将执行以下步骤：
- 使用输入数据 X 进行前向传播，得到预测值。
- 计算损失（预测值与实际值之间的差异）。
- 使用反向传播计算梯度。
- 更新模型参数（权重和偏置）。
我们将训练模型 1000 轮，并在每 100 轮打印一次损失。
```python
# 训练模型
num_epochs = 1000  # 训练 1000 轮
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式

    # 前向传播
    predictions = model(X)  # 模型输出预测值
    loss = criterion(predictions.squeeze(), Y)  # 计算损失（注意预测值需要压缩为1D）

    # 反向传播
    optimizer.zero_grad()  # 清空之前的梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新模型参数

    # 打印损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')
```

- predictions.squeeze()：我们在这里将模型的输出从 2D 张量压缩为 1D，因为目标值 Y 是一个一维数组。
- optimizer.zero_grad()：每次反向传播前需要清空之前的梯度。
- loss.backward()：计算梯度。
- optimizer.step()：更新权重和偏置。

## 模型评估

训练完成后，我们可以通过查看模型的权重和偏置来评估模型的效果。我们还可以在新的数据上进行预测并与实际值进行比较。
```python
# 查看训练后的权重和偏置
print(f'Predicted weight: {model.linear.weight.data.numpy()}')
print(f'Predicted bias: {model.linear.bias.data.numpy()}')

# 在新数据上做预测
with torch.no_grad():  # 评估时不需要计算梯度
    predictions = model(X)

# 可视化预测与实际值
plt.scatter(X[:, 0], Y, color='blue', label='True values')
plt.scatter(X[:, 0], predictions, color='red', label='Predictions')
plt.legend()
plt.show()
```

- model.linear.weight.data 和 model.linear.bias.data：这些属性存储了模型的权重和偏置。
- torch.no_grad()：在评估模式下，不需要计算梯度，节省内存。