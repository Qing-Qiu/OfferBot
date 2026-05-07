# 06 Matrix Derivatives：矩阵求导与常用梯度

## 核心直觉

矩阵求导的目标不是背复杂公式，而是稳定回答：

```text
loss 对每个参数的影响是什么
梯度 shape 是否和参数 shape 一致
这段反向传播怎么写成矩阵乘法
```

一个好习惯：每推一个梯度，都检查它的 shape 必须和被求导对象一样。

## 必要公式

线性回归：

$$
X\in\mathbb{R}^{n\times d},\qquad
w\in\mathbb{R}^{d\times 1},\qquad
y\in\mathbb{R}^{n\times 1}
$$

$$
\hat{y}=Xw+b,\qquad e=\hat{y}-y
$$

MSE：

$$
L=\frac{1}{n}e^\top e
$$

$$
\frac{\partial L}{\partial \hat{y}}=\frac{2}{n}e
$$

$$
\frac{\partial L}{\partial w}
=X^\top\frac{\partial L}{\partial \hat{y}}
=\frac{2}{n}X^\top(Xw+b-y)
$$

$$
\frac{\partial L}{\partial b}=\sum_i \frac{\partial L}{\partial \hat{y}_i}
$$

sigmoid：

$$
\sigma(z)=\frac{1}{1+\exp(-z)},\qquad
\sigma'(z)=\sigma(z)(1-\sigma(z))
$$

BCE：

$$
L=-\left[y\log p+(1-y)\log(1-p)\right]
$$

sigmoid + BCE：

$$
p=\sigma(z),\qquad \frac{\partial L}{\partial z}=p-y
$$

softmax：

$$
p_i=\frac{\exp(z_i)}{\sum_j \exp(z_j)}
$$

softmax + cross entropy：

$$
L=-\sum_i y_i\log p_i,\qquad
\frac{\partial L}{\partial z}=p-y
$$

线性层反向：

$$
Z=XW+b,\qquad G=\frac{\partial L}{\partial Z}
$$

$$
\frac{\partial L}{\partial W}=X^\top G,\qquad
\frac{\partial L}{\partial b}=\sum_{\text{batch}}G,\qquad
\frac{\partial L}{\partial X}=GW^\top
$$

BatchNorm / LayerNorm 核心统计量：

$$
\mu=\operatorname{average}(x),\qquad
\sigma^2=\operatorname{average}\left((x-\mu)^2\right)
$$

$$
\hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}},\qquad
y=\gamma \hat{x}+\beta
$$

差别直觉：BatchNorm 通常沿 batch 统计，LayerNorm 通常在单个样本的 hidden 维度上统计。

Attention score shape：

```text
Q: (B, H, T_q, D_h)
K: (B, H, T_k, D_h)
Q @ K.transpose(-2, -1): (B, H, T_q, T_k)
```

## AI 用法

线性回归、逻辑回归和神经网络线性层都共享 `XW + b` 这套梯度结构。

二分类中，直接使用 `BCEWithLogitsLoss` 比手写 `sigmoid + log` 更稳定，因为它会做数值稳定处理。

多分类中，`softmax + cross entropy` 的梯度简化成 `p - y`，这是分类模型训练最重要的公式之一。

Embedding lookup 的梯度可以理解为：只更新本 batch 查到的那些行，没查到的 embedding 行梯度为 0。

## 面试陷阱

- `dL/dw` 的 shape 必须和 `w` 一样。
- `X^T @ e` 中 `X^T.shape = (d, n)`，`e.shape = (n, 1)`，结果才是 `(d, 1)`。
- sigmoid 的导数不是 `sigmoid(x)`，而是 `sigmoid(x)(1-sigmoid(x))`。
- `softmax` 的 Jacobian 复杂，但和 cross entropy 合在一起后梯度很简单。
- 实战里不要先 softmax 再喂给 `CrossEntropyLoss`；PyTorch 的 `CrossEntropyLoss` 接收 logits。

## 自测题

1. `X.shape = (n, d)`，`w.shape = (d, 1)`，`Xw` 的 shape 是什么？
2. MSE 中 `e = y_hat - y`，为什么 `dL/dw = X^T @ e` 的 shape 是 `(d, 1)`？
3. 写出 sigmoid 的导数。
4. 为什么 sigmoid + BCE 的梯度可以简化成 `p - y`？
5. 多分类 softmax + cross entropy 对 logits 的梯度是什么？
6. `Z = XW + b`，`G = dL/dZ`，写出 `dL/dW`、`dL/db`、`dL/dX`。
7. 为什么 PyTorch 的 `CrossEntropyLoss` 不需要你先手动 softmax？
8. Embedding lookup 的梯度为什么是稀疏直觉？
9. BatchNorm 和 LayerNorm 的统计维度有什么区别？
10. Attention score 矩阵的最后两个维度分别表示什么？
