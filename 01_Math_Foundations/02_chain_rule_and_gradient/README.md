# 02 Chain Rule And Gradient：链式法则与梯度

## 核心直觉

导数回答的是：输入动一点，输出会怎么动。梯度回答的是：有很多输入变量时，往哪个方向动，函数上升最快。

深度学习训练本质上就是：

```text
先前向计算 loss
再用链式法则把 loss 对每个参数的影响传回去
最后沿负梯度方向更新参数
```

反向传播不是新的数学规则，它就是链式法则在计算图上的系统化应用。

## 必要公式

一元导数：

```text
f'(x) = df / dx
```

偏导数：

```text
partial f / partial x_i
```

梯度：

```text
grad f(x) = [partial f / partial x_1, ..., partial f / partial x_d]
```

一元链式法则：

```text
y = f(g(x))
dy/dx = dy/dg * dg/dx
```

多层链式法则：

```text
L -> y_hat -> z -> W
partial L / partial W
= partial L / partial y_hat
* partial y_hat / partial z
* partial z / partial W
```

梯度下降：

```text
theta <- theta - lr * grad_theta L
```

线性层局部梯度：

```text
z = XW + b
dL/dW = X^T @ dL/dz
dL/db = sum over batch(dL/dz)
dL/dX = dL/dz @ W^T
```

## AI 用法

在 PyTorch 里，`loss.backward()` 做的就是从 `loss` 节点开始，沿计算图反向应用链式法则，把梯度累积到参数的 `.grad` 上。

在神经网络里，激活函数、线性层、归一化层、loss 都是计算图上的节点。每个节点只需要知道自己的局部导数，整个模型的梯度就能由链式法则拼出来。

在深层网络中，如果每层局部梯度都小于 `1`，连乘后容易梯度消失；如果很多局部梯度大于 `1`，容易梯度爆炸。

## 面试陷阱

- 梯度方向是函数上升最快方向，优化 loss 时要走负梯度方向。
- `backward()` 默认会累积梯度，所以训练循环里通常要先 `zero_grad()`。
- 链式法则不是只适用于标量；向量函数里会出现 Jacobian，但面试中通常理解 shape 和局部梯度即可。
- 梯度为 `0` 不一定是全局最优，也可能是局部最优或鞍点。
- 反向传播算的是参数对 loss 的影响，不是模型输出对输入的解释本身。

## 自测题

1. `y = (3x + 1)^2`，用链式法则求 `dy/dx`。
2. `L = (y_hat - y)^2`，`y_hat = wx + b`，写出 `dL/dw`。
3. 为什么梯度下降要用 `theta - lr * grad`，而不是加号？
4. PyTorch 为什么每轮训练通常要调用 `optimizer.zero_grad()`？
5. 什么情况下容易出现梯度消失？
6. `z = XW + b`，若 `X.shape = (B, d_in)`，`W.shape = (d_in, d_out)`，`dL/dW` 的 shape 是什么？
7. 反向传播和链式法则是什么关系？
8. 梯度为 0 一定代表模型训练好了么？为什么？
