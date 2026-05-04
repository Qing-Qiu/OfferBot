# 02 Attention Deep Dive：Attention 深入

## 核心直觉

Attention 可以理解成一次可学习的信息检索：

```text
Query: 我现在想找什么
Key: 每个位置有什么索引特征
Value: 真正要被取回的信息
```

每个 query 和所有 key 做相似度，得到权重，再用这些权重对 value 加权求和。

## 关键概念

`Q/K/V`：由输入 hidden states 线性投影得到。Q 用来发问，K 用来匹配，V 用来提供内容。

`attention score`：`QK^T / sqrt(D_h)`，表示 query 对每个 key 的关注程度。

`softmax`：把 score 变成概率权重。

`mask`：把不允许看的位置变成极小值，使 softmax 后权重接近 0。

`MHA`：Multi-Head Attention，每个 head 有独立投影。

`MQA`：Multi-Query Attention，多个 query head 共享 K/V，降低 KV Cache。

`GQA`：Grouped-Query Attention，多组 query head 共享一组 K/V，是 MHA 和 MQA 的折中。

`FlashAttention`：通过分块和 IO 优化减少显存读写，保持精确 attention 结果的同时提升效率。

## 必要公式 / shape / 流程

基础 shape：

```text
X.shape = (B, T, D)
H = head 数
D_h = D / H
```

投影并拆 head：

```text
Q,K,V: (B, T, D)
-> (B, T, H, D_h)
-> (B, H, T, D_h)
```

score：

```text
Q @ K.transpose(-2, -1)
(B, H, T_q, D_h) @ (B, H, D_h, T_k)
-> (B, H, T_q, T_k)
```

scaled dot-product：

```text
scores = QK^T / sqrt(D_h)
weights = softmax(scores + mask)
output = weights @ V
```

复杂度：

```text
attention score: O(B * H * T^2 * D_h)
attention matrix memory: O(B * H * T^2)
```

这就是长上下文的主要瓶颈之一。

## 代码阅读提示

Attention 代码里最值得逐行标 shape：

```text
q_proj/k_proj/v_proj
reshape/view 到 head 维
transpose/permute
matmul 得到 scores
加 mask
softmax
乘 V
transpose 回来
out_proj
```

看到 `transpose(-2, -1)`，通常是在转最后两个维度，把 K 从 `(T_k, D_h)` 变成 `(D_h, T_k)`，以便和 Q 做矩阵乘法。

## 面试高频问法

1. Q、K、V 分别是什么意思？
2. 为什么 attention 要除以 `sqrt(D_h)`？
3. Attention 的时间复杂度和空间复杂度是多少？
4. Causal Mask 如何实现？
5. MHA、MQA、GQA 有什么区别？
6. KV Cache 缓存的是什么？
7. FlashAttention 解决了什么瓶颈？
8. 长上下文为什么难？

## 常见陷阱

- 忘记 `K.transpose(-2, -1)`，导致矩阵乘法维度不对。
- 把 mask 加在 softmax 后。mask 应该在 softmax 前作用到 logits/scores。
- 以为 FlashAttention 是近似 attention；它的核心是 IO 优化，不是简单近似。
- 只算时间复杂度，忘记 attention matrix 的显存压力。
- 混淆 MQA/GQA 的 K/V 共享方式。

## 自测题

1. Q、K、V 分别由什么得到？
2. `Q.shape=(B,H,T,D_h)`，`K.shape=(B,H,T,D_h)`，`QK^T` 的 shape 是什么？
3. 为什么 score 要除以 `sqrt(D_h)`？
4. causal mask 会把哪些位置屏蔽？
5. Attention 的 `T^2` 瓶颈来自哪里？
6. MQA 为什么能减少 KV Cache？
7. FlashAttention 的核心优化方向是什么？
8. `weights @ V` 的输出 shape 是什么？
