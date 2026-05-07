# 01 Linear Algebra Review：AI 高频线性代数

## 核心直觉

AI 里的线性代数不是为了做抽象证明，而是为了回答三类问题：

```text
shape 能不能对上
两个向量像不像
一个变换把数据压到哪里
```

向量可以看成一个样本的特征，也可以看成用户、物品、token 的 embedding。矩阵可以看成一批样本、一组参数，或者一个线性变换。张量只是把矩阵再加上 batch、head、time 等维度。

最重要的 shape 口诀：

```text
(m, k) @ (k, n) -> (m, n)
```

左边保留行数，右边保留列数，中间维度必须相等。

## 必要公式

点积：

$$
x \cdot w = \sum_i x_i w_i
$$

矩阵乘法：

$$
C = AB,\qquad C_{ij} = \sum_k A_{ik}B_{kj}
$$

转置、单位矩阵、逆矩阵：

$$
(A^\top)_{ij} = A_{ji},\qquad AI = IA = A,\qquad AA^{-1}=A^{-1}A=I
$$

秩：

$$
\operatorname{rank}(A)=\text{矩阵中线性无关行/列的最大数量}
$$

特征值与特征向量：

$$
Av = \lambda v
$$

含义：矩阵 `A` 作用到 `v` 后，`v` 的方向不变，只被拉伸或压缩了 `lambda` 倍。

SVD：

$$
A = U\Sigma V^\top
$$

含义：任意矩阵都可以拆成“旋转/反射 -> 缩放 -> 旋转/反射”的组合。奇异值越大，对应方向保留的信息越多。

线性模型：

```text
X.shape = (n, d)
w.shape = (d,) 或 (d, 1)
b.shape = 标量、(1,) 或 (1, 1)
y_hat = X @ w + b
y_hat.shape = (n,) 或 (n, 1)
```

L2 范数与欧氏距离：

$$
\|x\|_2 = \sqrt{\sum_i x_i^2},\qquad \operatorname{dist}(x,y)=\|x-y\|_2
$$

余弦相似度：

$$
\cos(x,y)=\frac{x\cdot y}{\|x\|_2\|y\|_2}
$$

投影：

$$
\operatorname{proj}_u(x)=\frac{x\cdot u}{u\cdot u}u
$$

PCA 的核心：

```text
找方差最大的方向
把数据投影到这些方向
保留主要信息，丢掉低方差噪声
```

Attention shape：

```text
Q.shape = (B, T_q, D)
K.shape = (B, T_k, D)
Q @ K^T -> (B, T_q, T_k)
```

多头注意力：

```text
Q.shape = (B, H, T_q, D_h)
K.shape = (B, H, T_k, D_h)
Q @ K.transpose(-2, -1) -> (B, H, T_q, T_k)
```

## AI 用法

在线性回归和神经网络里，`X @ W + b` 是最常见的线性层。`X` 是一批样本，`W` 把特征维度从 `d_in` 映射到 `d_out`。

在推荐系统里，用户向量 `u` 和物品向量 `v` 的点积常被当作匹配分数：

$$
\operatorname{score}(u,v)=u\cdot v
$$

在向量检索里，余弦相似度更关注方向，点积会同时受方向和向量长度影响。

在 Transformer 里，`Q @ K^T` 得到 token 对 token 的注意力分数矩阵。

PCA / SVD 常用于降维、去噪、矩阵分解、embedding 直觉和推荐系统早期模型理解。

## 面试陷阱

- 正交不是“不共线”，而是点积为 `0`，几何上垂直。
- `w.shape = (d,)` 和 `(d, 1)` 都常见，但输出 shape 不同。
- 点积不是严格的相似度；没有归一化时，向量长度会影响分数。
- PCA 不是随便选几个原特征，而是找新的正交方向。
- Attention 中 `K^T` 转的是最后两个维度，不是把 batch 维也转掉。

## 自测题

1. `X.shape = (128, 64)`，`W.shape = (64, 10)`，`X @ W` 的 shape 是什么？
2. `x = [1, 2, 3]`，`w = [4, 5, 6]`，`x · w` 等于多少？
3. `user_emb.shape = (B, D)`，`item_emb.shape = (B, D)`，逐样本匹配分数 shape 是什么？
4. `user_emb @ item_emb.T` 的 shape 是什么？第 `i, j` 个元素表示什么？
5. 为什么余弦相似度比点积更不受向量长度影响？
6. 两个向量正交时，它们的点积是多少？几何含义是什么？
7. PCA 想保留什么，丢掉什么？
8. `Q.shape = (B, H, T, D_h)`，`K.shape = (B, H, T, D_h)`，`Q @ K.transpose(-2, -1)` 的 shape 是什么？
