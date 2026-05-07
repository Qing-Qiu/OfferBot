# 03 Position Encoding：位置编码

## 核心直觉

Transformer 的 self-attention 很强，但它天然只看一组 token 向量之间的相似度。若不加入位置信息，模型很难区分：

```text
我 爱 你
你 爱 我
```

这两个序列的 token 集合相同，但顺序不同，含义完全不同。位置编码要解决的就是：让模型知道每个 token 在序列里的位置，以及 token 之间的相对距离。

从数学上看，位置编码就是把“第几个位置”变成向量、偏置或旋转，再注入 Attention 计算。

## 相关数学

位置编码主要关联五类数学知识：

1. 线性代数：embedding、向量加法、点积、矩阵乘法、shape。
2. 三角函数：sin、cos、周期、频率、相位。
3. 傅里叶直觉：用多种频率表达不同尺度的位置变化。
4. 旋转矩阵与复数：RoPE 把位置变成二维子空间里的旋转。
5. Attention 数学：位置最终影响 attention logits，也就是 `QK^T` 里的分数。

不需要先系统学完傅里叶分析，但要有这个直觉：不同频率的 sin / cos 可以让模型同时感知短距离和长距离。

## 关键概念

### 1. 为什么原始 Attention 不知道顺序

Self-attention 的核心分数是：

$$
\operatorname{score}_{ij}=q_i^\top k_j
$$

它只比较第 `i` 个 token 的 query 和第 `j` 个 token 的 key 是否匹配。如果输入 token embedding 没带位置信息，Attention 本身不会知道谁在前、谁在后。

所以 Transformer 需要额外注入位置。

### 2. 绝对位置编码

绝对位置编码给每个位置一个向量 `p_t`，然后和 token embedding 相加：

$$
h_t=e_t+p_t
$$

其中 `e_t` 是第 `t` 个 token 的语义向量，`p_t` 是第 `t` 个位置的向量。

最常见的两类绝对位置编码：

- learned position embedding：位置向量是可训练参数。
- sinusoidal position encoding：位置向量由 sin / cos 固定公式生成。

### 3. Sinusoidal Position Encoding

经典 Transformer 使用固定的正弦位置编码：

$$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/D}}\right)
$$

$$
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/D}}\right)
$$

这里：

```text
pos: token 在序列中的位置
i: embedding 维度中的频率编号
D: embedding 维度
```

偶数维用 `sin`，奇数维用 `cos`。不同维度对应不同频率，有的维度变化很快，适合表达局部位置；有的维度变化很慢，适合表达长距离位置。

### 4. 相对位置编码

绝对位置编码关心“这个 token 是第几个”。相对位置编码更关心：

```text
token i 和 token j 距离多远
token j 在 token i 的左边还是右边
```

一种常见做法是在 attention logits 里加入相对位置偏置：

$$
\operatorname{score}_{ij}=q_i^\top k_j+b_{i-j}
$$

其中 `b_{i-j}` 只和相对距离有关。

### 5. RoPE：旋转位置编码

RoPE 的核心直觉是：位置不再简单相加，而是让 query / key 按位置旋转。

二维旋转矩阵：

$$
R_\theta=
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
$$

把向量旋转后：

$$
\tilde{q}_m=R_mq_m,\qquad \tilde{k}_n=R_nk_n
$$

Attention 分数变成：

$$
\tilde{q}_m^\top \tilde{k}_n=(R_mq_m)^\top(R_nk_n)
$$

RoPE 的关键好处是：点积里会自然出现相对位置 `m-n` 的信息，所以它既保留了位置，又更适合建模相对距离。

如果你想把这条结论推清楚，继续看：[01_rope_deep_dive](./01_rope_deep_dive.md)。

### 6. ALiBi：线性位置偏置

ALiBi 不给 token 加位置向量，而是直接给 attention logits 加一个与距离相关的线性惩罚：

$$
\operatorname{score}_{ij}=q_i^\top k_j-m_h|i-j|
$$

其中 `m_h` 是第 `h` 个 head 的斜率。距离越远，分数惩罚越大。它简单、便宜，并且在长上下文外推上有一定优势。

## 必要公式 / shape / 流程

### Absolute PE shape

```text
token_emb:    (B, T, D)
position_emb: (T, D) 或 (1, T, D)
hidden = token_emb + position_emb
hidden:       (B, T, D)
```

### Attention 中位置的影响

如果位置编码加到输入 embedding：

$$
h_t=e_t+p_t
$$

后续 projection 会得到：

$$
q_t=h_tW_Q,\qquad k_t=h_tW_K,\qquad v_t=h_tW_V
$$

位置最终会进入：

$$
QK^\top
$$

也就是每个 token 对其他 token 的注意力分数。

### Relative bias shape

```text
attention_logits: (B, H, T, T)
relative_bias:    (H, T, T) 或 (1, H, T, T)
logits + bias:    (B, H, T, T)
```

### RoPE shape

```text
Q, K:        (B, H, T, D_h)
position:    (T,)
sin, cos:    (T, D_h / 2)
rotated Q,K: (B, H, T, D_h)
```

RoPE 通常把最后一维拆成二维小块，每一对维度做一次旋转。

## 代码阅读提示

看到 learned position embedding 时，重点看三件事：

```text
position_ids = arange(T)
position_emb = embedding(position_ids)
hidden = token_emb + position_emb
```

看到 sinusoidal PE 时，重点看：

```text
构造 position: (T, 1)
构造 div_term: (D/2,)
偶数维填 sin(position * div_term)
奇数维填 cos(position * div_term)
```

看到 RoPE 时，重点看：

```text
Q/K 的最后一维是否两两成对
是否预先缓存 sin/cos
是否只旋转 Q 和 K，不旋转 V
KV Cache 时新 token 的 position_id 是否正确
```

看到 relative position bias 时，重点看：

```text
bias 是否加在 softmax 之前
bias shape 是否能 broadcast 到 (B, H, T, T)
causal mask 是否仍然保留
```

## 面试高频问法

1. Transformer 为什么需要位置编码？
2. 绝对位置编码和相对位置编码有什么区别？
3. sinusoidal position encoding 为什么用不同频率的 sin / cos？
4. learned position embedding 有什么缺点？
5. RoPE 的核心思想是什么？
6. RoPE 为什么能表达相对位置？
7. RoPE 为什么只作用在 Q / K 上，而不是 V 上？
8. ALiBi 和 RoPE 的区别是什么？
9. 位置编码如何影响 Attention score？
10. 长上下文外推为什么和位置编码有关？

## 常见陷阱

- 位置编码不是只为了“告诉模型第几个”，更重要的是影响 token 之间的相对关系。
- 原始 self-attention 没有顺序感；顺序来自 token 表示或 attention logits 里的额外位置信息。
- learned absolute PE 对训练时没见过的更长序列外推较弱。
- sinusoidal PE 固定、不训练，但不代表一定比 learned PE 更好。
- RoPE 不是把位置向量加到 embedding 上，而是旋转 Q / K。
- Relative bias 要加在 softmax 之前；softmax 之后再加就不是改注意力分数了。
- KV Cache 场景下 position_id 不能每步都从 0 开始，否则 RoPE 位置信息会乱。

## 自测题

1. 为什么没有位置编码时，Transformer 很难区分“我爱你”和“你爱我”？
2. `token_emb.shape = (B, T, D)`，`position_emb.shape = (T, D)`，相加后 shape 是什么？
3. sinusoidal PE 中不同维度为什么要用不同频率？
4. 绝对位置编码和相对位置编码分别回答什么问题？
5. relative bias 应该加在 softmax 前还是后？为什么？
6. RoPE 的“旋转”发生在 Q/K/V 的哪几个张量上？
7. RoPE 为什么和二维旋转矩阵有关？
8. ALiBi 的线性距离惩罚对远距离 token 有什么影响？
9. KV Cache 解码时，为什么新 token 的 position_id 很重要？
10. 如果模型训练最大长度是 2048，直接推到 8192，位置编码可能带来什么问题？
