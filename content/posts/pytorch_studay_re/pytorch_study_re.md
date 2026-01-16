---
date: '2026-01-16T20:13:04+08:00'
draft: false
title: 'Pytorch学习重制版'
---

# Pytorch学习重制版前言

前面写过一个[Pytorch学习笔记](https://wwwtwilight.github.io/posts/pytorchstudy/pytorchstudy/)，现在回看写的就是一坨，所以决定重写一版，这一次是有参考教程的[《PyTorch深度学习实践》完结合集](https://www.bilibili.com/video/BV1Y7411d7Ys/)，这次重制版会更加系统化和规范化

目标是能够做到不使用AI自动生成代码，仅依靠自动补全就能完成代码编写

来几个小任务：
1. 实现一个线性回归，不会这个可以滚了，然后实现一个简单的神经网络
2. mnist手写数字识别，至少做到这里
3. 实现一个小的LLM？好吧这个牛逼好像吹得有点大，不过可以试试

# 线性模型&梯度下降

这部分原理都会，主要是代码，这里主要掌握了numpy的个别用法

```python
x_data = np.array([1.0, 2.0, 3.0]) # numpy声明数组
y_data = np.array([2.0, 4.0, 6.0])
xy_data = np.column_stack((x_data, y_data)) # column_stack函数将一维数组按列合并成二维数组
# 结果: [[1.0, 2.0],
#        [2.0, 4.0],
#        [3.0, 6.0]]

for w in np.arange(0.0, 10.0, 0.1): # np.arange函数可以创建包含浮点数的数组
    loss_sum = 0
    for x, y in xy_data:
        y_pred = forward(x, w)
        loss_sum += loss(y, y_pred)
    mse = loss_sum / len(xy_data)
    w_list.append(w)
    mse_list.append(mse)
```

# torch重要的数据类型——Tensor

Tensor在Pytorch中担任的是数组的角色，当然，随着维度的变化，可以是单个数，也可以是向量、矩阵，甚至更高维度的数组，其关键的一点在于满足torch的很多操作，比较重要的是自动存储梯度

```python
import torch
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True) # 创建一个一维Tensor，并且开启梯度追踪
y = x * 2  # 对Tensor进行操作
z = y.mean()  # 计算y的均值
z.backward()  # 反向传播，计算梯度
print(x.grad)  # 输出x的梯度
```

使用案例

```python
w = torch.tensor([[1.0]], requires_grad=True) # requires_grad=True使得w存储了梯度信息

for epoch in range(100):
    for x, y in xy_data:
        l = loss(forward(x, w), y) # 前向传播计算损失
        l.backward() # 反向传播计算梯度，这里使用的方式是每一个样本点都计算一次梯度并更新参数
        w.data = w.data - 0.01 * w.grad.data # 使用梯度下降法更新参数
        w.grad.data.zero_() # 清空梯度，避免梯度累加

for epoch in range(100):
    l_sum = 0
    for x, y in xy_data:
        l = loss(forward(x, w), y)
        l_sum += l # 累加每个样本点的损失
    l_sum.backward() # 对总损失进行反向传播计算梯度
    w.data = w.data - 0.01 * w.grad.data
    w.grad.data.zero_()
```

注意几个点
1. 计算图在前向传播过程中自动构建（原理是在每一个张量中记录计算过程，包括操作和依赖关系），pytorch会自动识别计算过程中是否包含requires_grad=True的张量，然后构建计算图
1. l.backward()过程中，会根据计算图自动计算梯度，但是不会更新参数，要手动更新参数
2. w.data访问的是张量的数值部分，w.grad访问的是梯度，梯度本身也是一个tensor，所以要使用w.grad.data才能访问到数值
3. 每次更新参数后要清空梯度，否则梯度会累加
4. 老师还讲到了item()函数，可以将只有一个元素的tensor转换为python的数值类型，但是在这里并没有用到
5. item和data，item是获取数值，data是获取tensor本身，data说明这次操作不需要梯度追踪

这部分内容还挺重要的，先写到这里，后面估计还会补充

