# 01 Transformer Basics：Transformer 基础

## 核心直觉

Transformer 的核心是：让序列里的每个 token 都能直接“看见”其他 token，并根据相关性加权汇总信息。

RNN 按时间一步步处理，天然串行；Transformer 用 Attention 一次性计算 token 之间的关系，因此训练时更容易并行。

一个 Transformer block 通常包含：

```text
token embedding
position information
self-attention
residual connection
LayerNorm
FFN
```

## 关键概念

`Tokenization`：把文本切成 token。常见思路包括 BPE、WordPiece、SentencePiece。

`Token Embedding`：把 token id 映射成向量。

`Position Encoding`：给模型位置信息，因为 Attention 本身不天然知道顺序。

`Self-Attention`：同一序列内部 token 互相计算相关性。

`Multi-Head Attention`：多个 attention head 并行学习不同关系。

`FFN`：逐 token 的前馈网络，通常负责非线性变换和特征升维/降维。

`Residual Connection`：保留输入信息，缓解深层训练困难。

`LayerNorm`：稳定每层激活分布。

`Causal Mask`：解码器里禁止当前位置看到未来 token。

`Padding Mask`：忽略 padding token。

## 必要公式 / shape / 流程

输入：

```text
token_ids.shape = (B, T)
embedding.shape = (B, T, D)
```

Q/K/V projection：

```text
Q = X W_Q
K = X W_K
V = X W_V
```

Self-Attention：

```text
Attention(Q,K,V) = softmax(QK^T / sqrt(D_h)) V
```

单层 block 流程：

```text
X
-> Multi-Head Self-Attention
-> Residual + LayerNorm
-> FFN
-> Residual + LayerNorm
```

Decoder-only LLM：

```text
输入前缀 tokens
使用 causal mask
每个位置预测下一个 token
```

## 代码阅读提示

读 Transformer 代码时，先找这些对象：

```text
embedding 层
position encoding / rotary embedding
attention projection: q_proj, k_proj, v_proj, o_proj
MLP/FFN: up/down/gate projection
norm: LayerNorm 或 RMSNorm
mask: causal mask / attention mask
```

再跟 shape：

```text
(B, T, D)
-> q/k/v projection
-> (B, H, T, D_h)
-> attention score (B, H, T, T)
-> output (B, T, D)
```

## 面试高频问法

1. Transformer 相比 RNN 为什么更容易并行？
2. 为什么需要位置编码？
3. Self-Attention 在做什么？
4. Multi-Head Attention 的意义是什么？
5. FFN 在 Transformer 里有什么作用？
6. Residual Connection 为什么重要？
7. LayerNorm 放在前面和后面有什么区别？
8. Causal Mask 和 Padding Mask 分别解决什么？

## 常见陷阱

- 以为 Attention 天然知道 token 顺序，忘记位置编码。
- 把 head 数量理解成越多越好，忽略每个 head 的维度和计算成本。
- 混淆 encoder、decoder、decoder-only 架构。
- 忘记 causal mask，导致训练时看到未来 token。
- 只讲 attention，不讲 FFN、残差、归一化这些稳定训练组件。

## 自测题

1. `token_ids.shape = (B,T)`，embedding 后 shape 是什么？
2. 为什么 Transformer 需要 position information？
3. Self-Attention 的输入输出 shape 通常是什么？
4. Multi-Head Attention 为什么要拆 head？
5. Causal Mask 在语言模型里防止什么？
6. Residual Connection 对深层网络有什么帮助？
7. LayerNorm 和 BatchNorm 在序列模型中为什么常选 LayerNorm？
8. Decoder-only LLM 的训练目标是什么？
