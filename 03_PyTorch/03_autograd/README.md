# 03 Autograd：自动微分

## 核心直觉

autograd 是 PyTorch 的反向传播引擎。你写前向计算，PyTorch 在背后记录计算图；你调用 `loss.backward()`，它沿计算图反向应用链式法则，把梯度算到参数的 `.grad` 上。

训练循环最小闭环：

```text
forward -> loss -> zero_grad -> backward -> optimizer.step
```

如果你能解释这五步，PyTorch 训练代码就已经读懂了一半。

## 关键概念

`requires_grad`：是否追踪这个 Tensor 的梯度。

叶子节点：通常是用户直接创建并需要优化的参数，例如 `nn.Parameter`。优化器主要更新叶子参数。

`grad_fn`：记录这个 Tensor 是由哪个操作产生的。非叶子节点通常有 `grad_fn`。

`.grad`：梯度累积的位置。参数的 `.grad` 在 `backward()` 后被填充或累加。

`backward()`：从标量 loss 出发反向传播。

`zero_grad()`：清空上一轮梯度，避免梯度跨 batch 累积。

`detach()`：从当前计算图切断，得到不再追踪历史梯度的 Tensor。

`torch.no_grad()`：上下文管理器，常用于评估和推理，避免构建计算图。

## 必要公式 / shape / 流程

计算图例子：

```text
x -> linear -> activation -> loss
```

反向传播：

```text
dL/dW = dL/dactivation * dactivation/dlinear * dlinear/dW
```

训练循环中的梯度生命周期：

```text
optimizer.zero_grad()
pred = model(x)
loss = criterion(pred, y)
loss.backward()
optimizer.step()
```

梯度累积：

```text
param.grad = param.grad + 当前 batch 梯度
```

所以不清零就会把多个 batch 的梯度加在一起。

detach 直觉：

```text
y = f(x)
z = y.detach()
```

`z` 的数值和 `y` 一样，但不再把梯度传回 `y` 之前的计算图。

## 代码阅读提示

读 autograd 代码时，先找这些线索：

```text
哪些 Tensor requires_grad=True
loss 是否是标量
backward 前有没有 zero_grad
训练和评估是否正确切换 no_grad
有没有 in-place 操作改坏计算图
有没有 detach 切断梯度
```

如果 loss 不是标量，PyTorch 需要你传入外部梯度，例如 `tensor.backward(torch.ones_like(tensor))`。日常训练里 loss 通常会被 reduce 成标量。

## 面试高频问法

1. PyTorch autograd 的基本原理是什么？
2. 什么是计算图？
3. 什么是叶子节点？
4. 为什么梯度会累积？
5. 为什么每轮要 `zero_grad()`？
6. `detach()` 和 `no_grad()` 有什么区别？
7. 评估阶段为什么要 `model.eval()` 和 `torch.no_grad()`？
8. in-place 操作为什么可能破坏 autograd？

## 常见陷阱

- 忘记 `zero_grad()`，导致梯度跨 batch 累积。
- 在 eval 阶段忘记 `no_grad()`，浪费显存。
- 把需要训练的 Tensor `detach()` 了，导致梯度断掉。
- 对参与反向传播的 Tensor 做危险 in-place 修改。
- 以为所有 Tensor 的 `.grad` 都会保留；默认主要保留叶子节点梯度。

## 自测题

1. `requires_grad=True` 的含义是什么？
2. `loss.backward()` 会把梯度写到哪里？
3. 为什么 PyTorch 默认累积梯度？
4. `detach()` 会改变 Tensor 数值吗？会改变梯度路径吗？
5. `torch.no_grad()` 常用于什么阶段？
6. 叶子节点和非叶子节点的区别是什么？
7. 为什么 loss 通常要是标量？
8. 如果训练 loss 不下降，你会从 autograd 角度排查哪些点？
