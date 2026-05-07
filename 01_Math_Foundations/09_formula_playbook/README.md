# 09 Formula Playbook：AI 高频公式速查表

## 核心直觉

这份速查表只保留面试和代码里最常出现的公式。复习时先看 shape，再看公式，再想它对应哪段代码。

## 必要公式

矩阵乘法：

```text
(m, k) @ (k, n) -> (m, n)
```

$$
C_{ij}=\sum_k A_{ik}B_{kj}
$$

线性层：

$$
Z=XW+b
$$

```text
X: (B, d_in)
W: (d_in, d_out)
b: (d_out,)
Z: (B, d_out)
```

线性层反向：

$$
G=\frac{\partial L}{\partial Z}
$$

$$
\frac{\partial L}{\partial W}=X^\top G,\qquad
\frac{\partial L}{\partial b}=\sum_{\text{batch}}G,\qquad
\frac{\partial L}{\partial X}=GW^\top
$$

MSE：

$$
L=\frac{1}{n}\sum_i(\hat{y}_i-y_i)^2,\qquad
\frac{\partial L}{\partial \hat{y}}=\frac{2}{n}(\hat{y}-y)
$$

sigmoid：

$$
\sigma(z)=\frac{1}{1+\exp(-z)},\qquad
\sigma'(z)=\sigma(z)(1-\sigma(z))
$$

BCE：

$$
L=-\left[y\log p+(1-y)\log(1-p)\right],\qquad
\text{sigmoid + BCE: }\frac{\partial L}{\partial z}=p-y
$$

softmax：

$$
p_i=\frac{\exp(z_i)}{\sum_j\exp(z_j)}
$$

softmax + cross entropy：

$$
L=-\sum_i y_i\log p_i,\qquad
\frac{\partial L}{\partial z}=p-y
$$

熵、交叉熵、KL：

$$
H(P)=-\sum_x P(x)\log P(x)
$$

$$
H(P,Q)=-\sum_x P(x)\log Q(x)
$$

$$
D_{\mathrm{KL}}(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}
$$

$$
H(P,Q)=H(P)+D_{\mathrm{KL}}(P\|Q)
$$

MLE / MAP：

$$
\operatorname{MLE}:\quad \arg\max_\theta \sum_i \log P(y_i\mid x_i;\theta)
$$

$$
\operatorname{MAP}:\quad \arg\max_\theta \left[\log P(D\mid\theta)+\log P(\theta)\right]
$$

梯度下降：

$$
\theta\leftarrow\theta-\eta\nabla_\theta L
$$

Attention：

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{D_h}}\right)V
$$

```text
Q,K,V: (B, H, T, D_h)
QK^T: (B, H, T, T)
output: (B, H, T, D_h)
```

FM 二阶项：

$$
\sum_{i<j}\langle v_i,v_j\rangle x_ix_j
$$

BPR：

$$
L=-\log\sigma(s_{\text{pos}}-s_{\text{neg}})
$$

NDCG：

$$
\operatorname{DCG}@K=\sum_{i=1}^{K}\frac{\operatorname{rel}_i}{\log_2(i+1)},\qquad
\operatorname{NDCG}@K=\frac{\operatorname{DCG}@K}{\operatorname{IDCG}@K}
$$

LoRA：

$$
W'=W+BA,\qquad \operatorname{rank}(BA)\le r
$$

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
