# 02 Tensor Basics：Tensor 基础

## 核心直觉

PyTorch Tensor 可以先理解成“能参与深度学习计算的 NumPy ndarray”。它不只是数组，还带着：

```text
shape: 数据长什么样
dtype: 每个数是什么类型
device: 数据放在 CPU 还是 GPU
requires_grad: 是否需要 autograd 追踪梯度
```

读 PyTorch 代码时，很多 bug 不是模型思想错了，而是 Tensor 的 shape、dtype、device 没对齐。Tensor 基础越稳，后面的模型代码越不吓人。

## 关键概念

`shape`：维度信息。例如 `(B, d)` 表示一个 batch 的特征，`(B, T, D)` 表示序列 embedding。

`dtype`：数据类型。模型参数常用 `float32` / `float16` / `bfloat16`，类别标签常用 `long`。

`device`：计算设备。常见是 `cpu`、`cuda`。参与同一次计算的 Tensor 通常必须在同一个 device 上。

`view` / `reshape`：改变 shape。`view` 要求内存布局连续，`reshape` 更灵活，必要时会拷贝。

`permute` / `transpose`：交换维度顺序。Attention 代码里非常常见。

`broadcast`：小 shape 自动扩展到大 shape。`(B, D) + (D,) -> (B, D)` 就是常见广播。

`contiguous`：Tensor 在内存里是否连续。`permute` 后常常不连续，后续 `view` 可能报错。

## 必要公式 / shape / 流程

常见 shape：

```text
单样本特征: (d,)
batch 特征: (B, d)
图像: (B, C, H, W)
序列 embedding: (B, T, D)
多头注意力: (B, H, T, D_h)
分类 logits: (B, C)
```

矩阵乘法：

```text
(B, d_in) @ (d_in, d_out) -> (B, d_out)
```

广播例子：

```text
x.shape = (B, D)
b.shape = (D,)
x + b -> (B, D)
```

reshape 读法：

```text
x.reshape(B, T, H, D_h)
```

通常表示把最后一维拆成多个 head，前提是元素总数不变。

permute 读法：

```text
x.permute(0, 2, 1, 3)
```

表示新维度顺序来自旧维度的第 `0, 2, 1, 3` 维。

## 代码阅读提示

读 Tensor 代码时，每一行都可以问：

```text
这行之前 shape 是什么
这行之后 shape 是什么
dtype 是否符合 loss 要求
device 是否和模型参数一致
是否需要 contiguous
```

分类任务里尤其注意：

```text
CrossEntropyLoss:
  logits: (B, C), float
  target: (B,), long

BCEWithLogitsLoss:
  logits: (B,) 或 (B, 1), float
  target: 同 shape, float
```

## 面试高频问法

1. Tensor 和 NumPy ndarray 有什么区别？
2. `view` 和 `reshape` 有什么区别？
3. `permute` 和 `transpose` 有什么区别？
4. 什么是 broadcasting？
5. 什么情况下需要 `.contiguous()`？
6. 为什么模型和数据必须在同一个 device？
7. `float32`、`float16`、`bfloat16` 有什么工程影响？
8. PyTorch 分类标签为什么常用 `long`？

## 常见陷阱

- CPU Tensor 和 GPU Tensor 混算。
- logits 是 float，但 target dtype 写错。
- 把 `(B,)` 和 `(B, 1)` 混用导致广播出意外 shape。
- `permute` 后直接 `view` 报错，因为内存不连续。
- 用 `reshape` 掩盖了错误的维度理解，结果模型能跑但语义错。

## 自测题

1. `x.shape = (32, 128)`，`W.shape = (128, 10)`，`x @ W` 的 shape 是什么？
2. `x.shape = (B, T, D)`，想变成 `(B, H, T, D_h)`，通常需要满足什么关系？
3. `CrossEntropyLoss` 的 logits 和 target shape 通常是什么？
4. 为什么 `permute` 后可能需要 `.contiguous()`？
5. `x.shape = (8, 16)`，`b.shape = (16,)`，`x + b` 的结果 shape 是什么？
6. `view` 和 `reshape` 哪个对连续内存更敏感？
7. 为什么 device mismatch 会报错？
8. 多头注意力里 `(B, H, T, D_h)` 每个维度分别表示什么？
