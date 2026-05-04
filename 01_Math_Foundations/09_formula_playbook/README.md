# 09 Formula Playbook：AI 高频公式速查表

## 核心直觉

这份速查表只保留面试和代码里最常出现的公式。复习时先看 shape，再看公式，再想它对应哪段代码。

## 必要公式

矩阵乘法：

```text
(m, k) @ (k, n) -> (m, n)
C_ij = sum_k A_ik B_kj
```

线性层：

```text
Z = XW + b
X: (B, d_in)
W: (d_in, d_out)
b: (d_out,)
Z: (B, d_out)
```

线性层反向：

```text
G = dL/dZ
dL/dW = X^T @ G
dL/db = sum over batch(G)
dL/dX = G @ W^T
```

MSE：

```text
L = (1/n) sum_i (y_hat_i - y_i)^2
dL/dy_hat = (2/n)(y_hat - y)
```

sigmoid：

```text
sigma(z) = 1 / (1 + exp(-z))
sigma'(z) = sigma(z)(1 - sigma(z))
```

BCE：

```text
L = -[y log p + (1-y) log(1-p)]
sigmoid + BCE: dL/dz = p - y
```

softmax：

```text
p_i = exp(z_i) / sum_j exp(z_j)
```

softmax + cross entropy：

```text
L = -sum_i y_i log p_i
dL/dz = p - y
```

熵、交叉熵、KL：

```text
H(P) = -sum_x P(x) log P(x)
H(P, Q) = -sum_x P(x) log Q(x)
KL(P || Q) = sum_x P(x) log(P(x) / Q(x))
H(P, Q) = H(P) + KL(P || Q)
```

MLE / MAP：

```text
MLE: argmax_theta sum_i log P(y_i | x_i; theta)
MAP: argmax_theta [log P(D | theta) + log P(theta)]
```

梯度下降：

```text
theta <- theta - lr * grad_theta L
```

Attention：

```text
Attention(Q,K,V) = softmax(QK^T / sqrt(D_h)) V
Q,K,V: (B, H, T, D_h)
QK^T: (B, H, T, T)
output: (B, H, T, D_h)
```

FM 二阶项：

```text
sum_{i<j} <v_i, v_j> x_i x_j
```

BPR：

```text
L = -log sigmoid(score_pos - score_neg)
```

NDCG：

```text
DCG@K = sum_i=1^K rel_i / log2(i + 1)
NDCG@K = DCG@K / IDCG@K
```

LoRA：

```text
W' = W + BA
rank(BA) <= r
```

## AI 用法

- `XW+b` 是线性回归、逻辑回归、MLP、Transformer projection 的共同骨架。
- `p-y` 是二分类和多分类里最重要的分类梯度直觉。
- `H(P,Q)` 和 `KL(P||Q)` 是理解分类、语言模型、蒸馏和偏好优化的核心。
- `QK^T` 是理解 Attention shape、显存复杂度和 KV Cache 的入口。
- FM、BPR、AUC、NDCG 是推荐系统面试里的高频数学组合。

## 面试陷阱

- 公式背下来不够，必须能说清 shape。
- PyTorch `CrossEntropyLoss` 接收 logits，不接收手动 softmax 后的概率。
- KL 不对称，不是真正距离。
- AUC 看相对排序，不看概率校准。
- KV Cache 用空间换时间，但长上下文 attention 仍然有显存压力。

## 自测题

1. 写出线性层 `Z=XW+b` 的前向 shape 和三个反向梯度。
2. 为什么 sigmoid + BCE 的梯度是 `p-y`？
3. 为什么 softmax + cross entropy 的梯度也是 `p-y`？
4. 写出交叉熵和 KL 的关系式。
5. MLE 和交叉熵训练有什么关系？
6. Attention 中 `QK^T` 的 shape 是什么？
7. BPR loss 想优化什么相对顺序？
8. AUC 和 NDCG 分别关注排序的哪个方面？
9. LoRA 为什么叫低秩适配？
