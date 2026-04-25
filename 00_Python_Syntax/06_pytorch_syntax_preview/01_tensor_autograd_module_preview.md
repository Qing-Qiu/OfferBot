# 01 PyTorch 语法预备：从 NumPy 过渡到 Tensor、autograd、nn.Module

> 学习目标：先不讲完整深度学习训练，只把你以后读 PyTorch 代码最常见的 4 个对象看懂：`Tensor`、`requires_grad`、`backward()`、`nn.Module`。

这一课的定位是：

```text
把 NumPy 的 shape 直觉迁移到 PyTorch
```

如果你已经理解了：

- `ndarray`
- `shape`
- `X @ w + b`
- 广播

那么进入 PyTorch 时，只需要再多学几件事：

- `Tensor`
- `device`
- `requires_grad`
- `nn.Module`

## 1. PyTorch Tensor 是什么

最粗暴但很有用的理解：

```text
PyTorch Tensor = 带自动求导能力、可放到 GPU 的 ndarray 近亲
```

和 NumPy 的共同点：

- 都有 `shape`
- 都有切片
- 都能做矩阵乘法
- 都支持广播

PyTorch 比 NumPy 多出来的能力：

- 可以放到 GPU
- 可以记录计算图
- 可以自动反向传播
- 可以和 `nn.Module`、`optimizer` 联动训练模型

## 2. 最基本的 Tensor 创建

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
```

这时：

```python
x.shape
X.shape
```

就和 NumPy 很像。

## 3. `shape`、`dtype`、`device`

这三个属性以后你会频繁看。

### `shape`

```python
X.shape
```

表示张量形状。

### `dtype`

```python
X.dtype
```

表示数据类型，比如：

- `torch.float32`
- `torch.int64`

### `device`

```python
X.device
```

表示张量在哪个设备上：

- `cpu`
- `cuda`

读 PyTorch 代码时一定要有这个意识：

```text
shape 决定能不能算
device 决定能不能一起算
```

如果两个 Tensor 不在同一个 device 上，很多操作会直接报错。

## 4. Tensor 和 NumPy 的对应关系

NumPy：

```python
y = X @ w + b
```

PyTorch：

```python
y = X @ w + b
```

这行几乎一样。

也就是说，前向传播的 shape 逻辑并没有变。

真正的变化在于：

```python
w = torch.tensor([1.0, 2.0], requires_grad=True)
```

这表示：

```text
这个张量不仅参与计算，还要参与求梯度
```

## 5. `requires_grad` 是什么

这是 PyTorch 的核心标记之一。

```python
w = torch.tensor([1.0, 2.0], requires_grad=True)
```

意思是：

```text
PyTorch 需要跟踪这个张量参与过哪些运算，
以便后面根据 loss 自动求导。
```

官方教程里明确说明：如果 `requires_grad=True`，Tensor 会跟踪自己的创建历史；如果某个运算的输入里有需要梯度的 Tensor，输出通常也会继续带上梯度信息。[来源](https://docs.pytorch.org/tutorials/beginner/nlp/pytorch_tutorial.html)

## 6. `grad_fn` 是什么

看这个：

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2
```

这时：

```python
y.grad_fn
```

不再是 `None`，因为：

```text
y 不是用户手工“原地定义”的叶子张量，
而是某个运算的结果。
```

你可以把 `grad_fn` 理解成：

```text
这个张量知道自己是怎么算出来的
```

## 7. `backward()` 做了什么

假设：

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
```

这里：

- `x.shape == (3,)`
- `y` 是一个标量

然后：

```python
y.backward()
```

意思是：

```text
从这个标量 y 出发，
沿着计算图反向传播，
把对叶子张量 x 的梯度算出来，
并累积到 x.grad 里。
```

官方文档明确写了：`Tensor.backward()` 会把梯度累积到 leaf Tensor 的 `.grad` 上，所以通常需要在下一次迭代前清掉旧梯度。[来源](https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.backward.html)

## 8. 为什么要 `zero_grad()`

这点面试里很常问。

PyTorch 默认不是覆盖梯度，而是：

```text
累积梯度
```

也就是说：

```python
loss.backward()
loss.backward()
```

如果中间不清空，`.grad` 会叠加。

官方教程也提醒了这一点：多次运行时，梯度会 increment，因为 PyTorch 会把梯度累积进 `.grad` 属性里。[来源](https://docs.pytorch.org/tutorials/beginner/nlp/pytorch_tutorial.html)

所以训练循环里经常看到：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## 9. `nn.Module` 是什么

`nn.Module` 是 PyTorch 组织模型的标准方式。

你可以把它理解成：

```text
一个会自动登记参数、支持 forward、支持 parameters() 的模型壳子
```

最小例子：

```python
import torch.nn as nn

class TinyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)
```

这段代码里：

- `self.linear` 是一个子模块
- 它内部有权重和偏置
- 这些参数会自动被 `model.parameters()` 收集到

官方教程明确提到：`nn.Module` 可以跟踪状态，并提供 `.parameters()`、`.zero_grad()` 等方法，适合构造模型类。[来源](https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html)

## 10. `forward()` 和 `model(x)` 的关系

在 PyTorch 里你通常写：

```python
pred = model(x)
```

而不是：

```python
pred = model.forward(x)
```

原因是：

```text
调用 model(x) 时，PyTorch 会自动走 Module 的调用逻辑，
内部再触发 forward。
```

所以读代码时记住：

```text
model(x) 本质上是在执行 forward
```

## 11. 最小训练三连

以后你最常见的训练语法就是：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

这三行含义分别是：

### `optimizer.zero_grad()`

清空旧梯度。

### `loss.backward()`

根据当前 loss 反向传播，算出参数梯度。

### `optimizer.step()`

根据已经算好的梯度更新参数。

## 12. 你现在应该怎么把 PyTorch 和 NumPy 连起来

NumPy 世界里你看到：

```python
y = X @ w + b
```

PyTorch 世界里你看到：

```python
y = X @ w + b
loss = ((y - target) ** 2).mean()
loss.backward()
```

你应该意识到：

```text
前向 shape 逻辑没变
只是在 PyTorch 里多了“自动求导”
```

这就是从 NumPy 迁移到 PyTorch 的关键桥梁。

## 13. 本节最该记住的 6 句话

1. `Tensor` 是 PyTorch 的基本数据结构。
2. `Tensor` 在 shape 逻辑上和 NumPy 很像。
3. `requires_grad=True` 表示这个 Tensor 要参与梯度计算。
4. `backward()` 会把梯度累积到叶子张量的 `.grad` 上。
5. `nn.Module` 是组织模型的标准写法。
6. `model(x)` 本质上是在调用 `forward()`。

## 14. 下一步

下一节最自然的延续是：

```text
写一个最小可运行的 Tensor + autograd + nn.Module demo
```

这样你就会第一次把：

- Tensor
- requires_grad
- loss.backward()
- model.parameters()

连成一条线。

## 参考资料

- [PyTorch Introduction to PyTorch](https://docs.pytorch.org/tutorials/beginner/nlp/pytorch_tutorial.html)
- [PyTorch What is torch.nn really?](https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html)
- [torch.Tensor.backward](https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.backward.html)

