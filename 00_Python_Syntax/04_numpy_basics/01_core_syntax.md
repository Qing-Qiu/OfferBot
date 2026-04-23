# 01 NumPy 基础语法：读懂 `shape`、`reshape`、`X @ w` 和广播

> 学习目标：把你在机器学习和 PyTorch 代码里最常见、最容易卡住的 NumPy 语法一次讲清楚。

这一课只讲 **语法和数据形状**，不讲机器学习原理。

也就是说：

- 在这里，你学的是 `ndarray`、`shape`、切片、矩阵乘法、广播。
- 在 `02_ML_Foundations` 里，你学的是线性回归、损失函数、梯度下降。

## 1. NumPy 到底是什么

NumPy 是 Python 里最常用的数值计算库。

它的核心对象是：

```python
ndarray
```

你可以把它理解成：

```text
支持多维形状、切片、矩阵运算、广播的高效数组
```

和 Python 原生 `list` 的区别：

| 类型 | 例子 | 适合做什么 |
| --- | --- | --- |
| `list` | `[1, 2, 3]` | 通用容器 |
| `np.ndarray` | `np.array([1, 2, 3])` | 数值计算、矩阵运算 |

## 2. 创建数组

### 最常见：`np.array`

```python
import numpy as np

x = np.array([1, 2, 3])
```

这时：

```python
type(x)   # numpy.ndarray
x.shape   # (3,)
```

### 常见创建方式

```python
np.zeros((2, 3))
np.ones((2, 3))
np.arange(0, 6)
np.linspace(0, 1, 5)
```

含义：

- `zeros((2, 3))`：创建 2 行 3 列全 0 数组。
- `ones((2, 3))`：创建 2 行 3 列全 1 数组。
- `arange(0, 6)`：生成 `[0, 1, 2, 3, 4, 5]`。
- `linspace(0, 1, 5)`：在 0 到 1 之间均匀取 5 个点。

## 3. `shape`、`ndim`、`dtype`

这是读 NumPy 代码最重要的三件事。

### `shape`

表示数组每一维的长度。

```python
x = np.array([1, 2, 3])
x.shape
```

结果：

```python
(3,)
```

说明它是一维数组，长度为 3。

```python
X = np.array([[1, 2, 3], [4, 5, 6]])
X.shape
```

结果：

```python
(2, 3)
```

说明它有 2 行 3 列。

### `ndim`

表示维度数。

```python
x.ndim   # 1
X.ndim   # 2
```

### `dtype`

表示元素类型。

```python
x.dtype
```

常见结果：

```python
int64
float64
```

在深度学习里，`dtype` 很重要，因为它会影响：

- 计算精度
- 显存占用
- 是否能和 PyTorch Tensor 对齐

## 4. 一维数组和二维数组的区别

这点是初学者最容易混乱的地方。

```python
x = np.array([1, 2, 3])
x.shape
```

结果：

```python
(3,)
```

这是一维数组，不是“1 行 3 列”。

```python
x_col = np.array([[1], [2], [3]])
x_col.shape
```

结果：

```python
(3, 1)
```

这是二维列向量。

```python
x_row = np.array([[1, 2, 3]])
x_row.shape
```

结果：

```python
(1, 3)
```

这是二维行向量。

所以：

```text
(3,)    != (3, 1) != (1, 3)
```

虽然看起来都像“3 个数”，但形状不同，很多运算结果也不同。

## 5. `reshape`

`reshape` 用来改形状，不改数据本身。

```python
x = np.array([1, 2, 3, 4])
x.shape
```

结果：

```python
(4,)
```

```python
x2 = x.reshape(2, 2)
x2.shape
```

结果：

```python
(2, 2)
```

### 最常见写法：`reshape(-1, 1)`

```python
x = np.array([1, 2, 3, 4])
x = x.reshape(-1, 1)
```

结果：

```python
x.shape == (4, 1)
```

这里的 `-1` 表示：

```text
这一维让我自动推断
```

这是把一维数组变成列向量的经典写法。

## 6. 索引和切片

### 一维数组

```python
x = np.array([10, 20, 30, 40])

x[0]    # 10
x[-1]   # 40
x[:2]   # [10, 20]
```

### 二维数组

```python
X = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])
```

```python
X[0]        # 第一行
X[:, 0]     # 第一列
X[:2, :]    # 前两行，所有列
X[:, 1:3]   # 所有行，第 2 到第 3 列
```

读这种语法时记住：

```text
X[行, 列]
```

其中：

- `:` 表示这一维全要。
- `1:3` 表示左闭右开，取第 1 到第 2 个位置。

## 7. 矩阵乘法：`@`

NumPy 里：

```python
y = X @ w
```

表示矩阵乘法。

这在机器学习里极其常见。

### 最重要的一个例子

```python
X.shape = (n_samples, n_features)
w.shape = (n_features,)
```

那么：

```python
y = X @ w
y.shape == (n_samples,)
```

直觉上：

```text
X 的每一行代表一个样本
w 是一组特征权重
每一行和 w 做点积，得到一个预测值
```

这就是线性回归里的：

```python
y_hat = X @ w + b
```

### 点积和矩阵乘法的关系

如果：

```python
x = np.array([1, 2, 3])
w = np.array([4, 5, 6])
```

那么：

```python
x @ w
```

结果是：

```python
1*4 + 2*5 + 3*6 = 32
```

也就是点积。

## 8. 广播机制

广播是 NumPy 很强大、也很容易出 Bug 的地方。

先看：

```python
X = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
b = np.array([10, 20, 30])

Y = X + b
```

结果：

```python
[
    [11, 22, 33],
    [14, 25, 36],
]
```

为什么可以加？

因为 `b.shape == (3,)`，NumPy 会自动把它理解成：

```text
把这一行复制到每一行上
```

所以：

```text
(2, 3) + (3,) -> (2, 3)
```

这就是广播。

### 在线性模型里的广播

```python
y_hat = X @ w + b
```

其中：

- `X @ w` 的形状通常是 `(n_samples,)`
- `b` 是一个标量

于是 NumPy 会自动把这个标量加到每个样本的预测值上。

## 9. 聚合操作

常见聚合：

```python
np.sum(x)
np.mean(x)
np.max(x)
np.argmax(x)
```

例子：

```python
x = np.array([1, 7, 3])

np.sum(x)      # 11
np.mean(x)     # 3.666...
np.max(x)      # 7
np.argmax(x)   # 1
```

其中：

- `argmax` 返回最大值的位置，不是最大值本身。

## 10. 怎么读 `X @ w + b`

这行代码你以后会反复看到。

```python
y_hat = X @ w + b
```

你应该这样拆：

### 第一步：先看类型

```text
X 是 ndarray
w 是 ndarray
b 是标量或 ndarray
```

### 第二步：再看 shape

```text
X.shape = (n_samples, n_features)
w.shape = (n_features,)
```

### 第三步：判断运算

```text
X @ w       -> 每个样本得到一个分数
+ b         -> 给每个分数加偏置
```

### 第四步：看输出 shape

```text
y_hat.shape = (n_samples,)
```

这就是读 NumPy / PyTorch 代码时最重要的习惯：

```text
类型 -> shape -> 运算 -> 输出 shape
```

## 11. 常见坑

### 坑 1：把 `(n,)` 当成 `(n, 1)`

这两个形状完全不同，很多矩阵运算结果也不同。

### 坑 2：不会看 `X[:, 0]`

这表示“所有行的第 0 列”。

### 坑 3：把 `argmax` 当最大值

`argmax` 返回位置，不返回值。

### 坑 4：广播没看 shape

广播很方便，但你必须知道它是怎么自动扩展的。

### 坑 5：不知道 `@` 是矩阵乘法

不是普通逐元素相乘。

逐元素相乘是：

```python
x * w
```

矩阵乘法是：

```python
x @ w
```

## 12. 这节课学完你至少要会

- 认出 `ndarray`
- 看懂 `shape`、`ndim`、`dtype`
- 分清 `(n,)`、`(n, 1)`、`(1, n)`
- 会读 `reshape(-1, 1)`
- 会读 `X[:, 0]`、`X[:5]`
- 会读 `X @ w`
- 会读简单广播
- 会读 `sum / mean / max / argmax`

## 13. 下一步衔接

学完这节，最自然的下一步是两条路：

1. 回到 `02_ML_Foundations` 重新看线性回归里的 `X @ w + b`
2. 继续学 `PyTorch Tensor`，因为很多语法和 NumPy 非常像
