# 02 shape 与矩阵乘法专项：把 `X @ w + b` 彻底看懂

> 学习目标：从“知道 NumPy 有 `shape` 和 `@`”提升到“能一眼看懂线性模型和 PyTorch 张量代码里的维度关系”。

这一课只做一件事：

```text
把 shape 看懂
```

因为后面你不管看：

- 线性回归
- MLP
- Embedding
- Attention
- batch 训练

本质都离不开：

```text
输入 shape 是什么？
输出 shape 是什么？
这一步为什么能乘？
这一步为什么能加？
```

## 1. 你以后看代码的固定顺序

遇到一行 NumPy / PyTorch 代码，永远先做这四步：

```text
1. 看类型
2. 看 shape
3. 看运算
4. 看输出 shape
```

例如：

```python
y_hat = X @ w + b
```

你应该这样读：

```text
X: (n_samples, n_features)
w: (n_features,)
b: 标量

X @ w -> (n_samples,)
+ b   -> (n_samples,)
```

## 2. `(n,)`、`(n, 1)`、`(1, n)` 必须强行分清

这是 NumPy 最容易出错的地方。

### 一维向量

```python
x = np.array([1, 2, 3])
x.shape == (3,)
```

### 列向量

```python
x_col = np.array([[1], [2], [3]])
x_col.shape == (3, 1)
```

### 行向量

```python
x_row = np.array([[1, 2, 3]])
x_row.shape == (1, 3)
```

它们不是一回事。

## 3. 为什么 `X @ w` 可以算

假设：

```python
X.shape = (4, 3)
w.shape = (3,)
```

那么：

```python
X @ w
```

合法，因为：

```text
X 的最后一维 = 3
w 的唯一一维 = 3
```

结果：

```text
(4, 3) @ (3,) -> (4,)
```

直觉：

```text
4 个样本
每个样本 3 个特征
w 给 3 个特征各一个权重
每个样本和 w 做一次点积
所以得到 4 个输出
```

## 4. 为什么 `X @ w + b` 也可以算

如果：

```python
(X @ w).shape == (4,)
b 是标量
```

那么：

```python
X @ w + b
```

合法，因为标量会广播到每个元素。

结果：

```text
(4,) + scalar -> (4,)
```

## 5. 如果 `w` 变成列向量会怎样

假设：

```python
X.shape = (4, 3)
w.shape = (3, 1)
```

那么：

```python
X @ w
```

结果：

```text
(4, 3) @ (3, 1) -> (4, 1)
```

这和 `(4,)` 不一样。

所以以后看到代码时要特别注意：

```text
同样是 3 个权重，
(3,) 和 (3, 1) 会导致输出 shape 不同。
```

## 6. 二维矩阵加一维向量：广播最经典场景

```python
X.shape = (2, 3)
b.shape = (3,)
Y = X + b
```

结果：

```text
(2, 3) + (3,) -> (2, 3)
```

直觉：

```text
b 被当成“每一行都加一次”
```

## 7. 你必须会口算的 6 个 shape

### 练习 1

```text
(5, 3) @ (3,) -> ?
答案：(5,)
```

### 练习 2

```text
(5, 3) @ (3, 1) -> ?
答案：(5, 1)
```

### 练习 3

```text
(5,) + scalar -> ?
答案：(5,)
```

### 练习 4

```text
(2, 4) + (4,) -> ?
答案：(2, 4)
```

### 练习 5

```text
(2, 4) + (2,) -> ?
答案：通常不合法，最后一维 4 和 2 对不上
```

### 练习 6

```text
(8, 16) @ (16, 32) -> ?
答案：(8, 32)
```

## 8. 线性回归代码里最重要的 3 行

```python
y_pred = X @ w + b
errors = y_pred - y
loss = np.mean(errors ** 2)
```

你应该这样读：

### 第 1 行

```text
X: (n_samples, n_features)
w: (n_features,)
b: scalar

y_pred: (n_samples,)
```

### 第 2 行

```text
y_pred: (n_samples,)
y:      (n_samples,)

errors: (n_samples,)
```

### 第 3 行

```text
errors ** 2: (n_samples,)
mean(...)   : scalar
```

这就是为什么 loss 最后会变成一个标量。

## 9. 常见错法

### 错法 1：不知道输出为什么从二维变成一维

```python
(n_samples, n_features) @ (n_features,) -> (n_samples,)
```

不是 `(n_samples, 1)`。

### 错法 2：广播没看最后一维

你不能只看“元素个数差不多”，要看能不能按 NumPy 广播规则对齐。

### 错法 3：把逐元素乘法当矩阵乘法

```python
X * w   # 逐元素
X @ w   # 矩阵乘法
```

这是两回事。

## 10. 这节学完后你该会什么

- 看见一行代码先报出 shape
- 分清 `(n,)`、`(n, 1)`、`(1, n)`
- 会判断 `@` 能不能算
- 会判断 `+ b` 是不是广播
- 能读懂线性回归前向传播和 loss 计算

## 11. 下一步

下一节最自然的衔接是：

```text
PyTorch Tensor 预备
```

因为 PyTorch 的很多形状思路和 NumPy 是共通的，只是多了：

- `device`
- `requires_grad`
- `nn.Module`

