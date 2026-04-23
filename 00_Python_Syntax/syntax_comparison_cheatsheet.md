# Python / NumPy / pandas / PyTorch 语法速查表

这份速查表的目标不是让你背语法，而是让你在读 AI 代码时快速判断：

```text
这一行到底是在操作普通 Python 对象、NumPy 数组、pandas 表格，还是 PyTorch Tensor？
```

## 1. 导入方式

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
```

含义：

- `np` 通常表示 NumPy，用于矩阵和数组计算。
- `pd` 通常表示 pandas，用于表格数据处理。
- `torch` 用于张量计算和自动微分。
- `nn` 通常表示 `torch.nn`，用于神经网络层和损失函数。

## 2. 创建数据

### Python list

```python
x = [1, 2, 3]
```

特点：灵活，但不适合大规模数值计算。

### NumPy ndarray

```python
x = np.array([1, 2, 3])
```

特点：适合 CPU 上的向量化数值计算。

### pandas DataFrame

```python
df = pd.DataFrame({
    "user_id": [1, 2, 3],
    "click": [1, 0, 1],
})
```

特点：适合行列表格、数据清洗、统计聚合。

### PyTorch Tensor

```python
x = torch.tensor([1.0, 2.0, 3.0])
```

特点：适合深度学习，可放到 GPU，可记录梯度。

## 3. 查看形状

```python
len(py_list)       # Python list 长度
arr.shape          # NumPy 数组形状
df.shape           # pandas 表格形状，返回 (行数, 列数)
tensor.shape       # PyTorch Tensor 形状
```

在 AI 代码里，`shape` 是最重要的调试线索之一。读不懂代码时，先问：

```text
这一行输入 shape 是什么？
这一行输出 shape 是什么？
```

## 4. 索引和切片

### Python list

```python
x[0]       # 第一个元素
x[-1]      # 最后一个元素
x[:3]      # 前三个元素
```

### NumPy

```python
X[0]       # 第一行
X[:, 0]    # 第一列
X[:5, :]   # 前五行、所有列
X[mask]    # 按布尔条件筛选行
```

### pandas

```python
df["click"]                  # 选择一列，返回 Series
df[["user_id", "click"]]     # 选择多列，返回 DataFrame
df[df["click"] == 1]         # 条件筛选行
df.loc[0, "click"]           # 按标签取值
df.iloc[0, 1]                # 按位置取值
```

### PyTorch

```python
x[0]
X[:, 0]
X[:5]
X[mask]
```

PyTorch Tensor 的很多切片语法和 NumPy 很像。

## 5. 矩阵乘法

```python
y = X @ w
```

这是 Python 里的矩阵乘法运算符，常用于 NumPy 和 PyTorch。

如果：

```text
X.shape = (n_samples, n_features)
w.shape = (n_features,)
```

那么：

```text
(X @ w).shape = (n_samples,)
```

线性回归中的预测：

```python
y_hat = X @ w + b
```

含义：

```text
每个样本的一行特征，和权重向量做点积，再加偏置。
```

## 6. reshape 类语法

### NumPy

```python
x.reshape(-1, 1)
```

常见含义：把一维数组变成二维列向量。

```text
原来: shape = (n,)
之后: shape = (n, 1)
```

### PyTorch

```python
x.view(-1, 1)
x.reshape(-1, 1)
```

`-1` 的意思是：这一维让系统根据元素总数自动推断。

## 7. 聚合计算

### NumPy

```python
np.mean(x)
np.sum(x)
np.max(x)
np.argmax(x)
```

### pandas

```python
df["click"].mean()
df.groupby("user_id")["click"].sum()
```

### PyTorch

```python
torch.mean(x)
torch.sum(x)
torch.argmax(logits, dim=1)
```

注意：PyTorch 里经常有 `dim` 参数，表示沿哪个维度计算。

## 8. 广播机制

广播就是形状不同的数组或张量，在规则允许时自动扩展后再计算。

```python
X + b
```

如果：

```text
X.shape = (100, 3)
b.shape = (3,)
```

那么 `b` 会被当成每一行都加一次。

广播很方便，但也是 Bug 高频来源。看到广播代码时要主动检查 shape。

## 9. 函数和方法的区别

函数写法：

```python
np.mean(x)
torch.mean(x)
```

方法写法：

```python
x.mean()
df.head()
model.fit(X, y)
```

直觉：

- 函数：把对象作为参数传进去。
- 方法：对象自己调用自己的能力。

## 10. 类和 self

```python
class Model:
    def __init__(self, weight):
        self.weight = weight

    def predict(self, x):
        return x * self.weight
```

`self` 表示“当前这个对象自己”。

调用时：

```python
model = Model(weight=3)
model.predict(10)
```

Python 会自动把 `model` 作为 `self` 传入 `predict`。

## 11. PyTorch 训练循环最常见语法

```python
for x, y in dataloader:
    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

逐行解释：

- `for x, y in dataloader`：每次取一个 batch 的输入和标签。
- `pred = model(x)`：调用模型的 `forward`，得到预测。
- `loss = loss_fn(pred, y)`：计算预测和标签之间的损失。
- `optimizer.zero_grad()`：清空旧梯度。
- `loss.backward()`：反向传播，计算新梯度。
- `optimizer.step()`：根据梯度更新参数。

## 12. 读代码时的三连问

遇到看不懂的一行代码，先问三个问题：

```text
1. 这个变量是什么类型？list、ndarray、DataFrame，还是 Tensor？
2. 这个变量的 shape 是什么？
3. 这一行是在改数据、算预测、算 loss，还是更新参数？
```

这三个问题能解决 70% 的 AI 代码阅读障碍。

