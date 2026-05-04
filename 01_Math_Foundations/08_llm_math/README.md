# 08 LLM Math：大模型数学

## 核心直觉

大模型训练的核心是概率建模：

```text
给定前面的 token，预测下一个 token 的概率分布
```

推理时，模型不是直接吐出唯一答案，而是在 vocabulary 上给出 logits，再经过 softmax 和采样策略选出下一个 token。

Transformer 的核心计算是 attention：每个 token 根据 query 去看其他 token 的 key，并加权汇总 value。

## 必要公式

自回归分解：

```text
P(x_1, ..., x_T) = product_t P(x_t | x_<t)
```

语言模型 NLL：

```text
L = -sum_t log P(x_t | x_<t)
```

softmax：

```text
p_i = exp(z_i) / sum_j exp(z_j)
```

temperature：

```text
p_i = softmax(z_i / tau)
```

Attention：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(D_h)) V
```

Shape：

```text
Q, K, V: (B, H, T, D_h)
QK^T: (B, H, T, T)
Attention output: (B, H, T, D_h)
```

KV Cache 直觉：

```text
prefill: 一次处理 prompt
decode: 每步只算新 token 的 Q，复用历史 K/V
```

LoRA：

```text
W' = W + Delta W
Delta W = B A
rank(Delta W) <= r
```

KL penalty：

```text
KL(pi_new || pi_ref)
```

用于约束新模型不要偏离参考模型太远。

量化误差：

```text
W_q = round(W / s) * s
error = W - W_q
```

直觉：用更低精度表示权重会节省显存和带宽，但会引入近似误差。

## AI 用法

预训练时，LLM 用 next-token prediction 学语言和世界知识。SFT 仍然常用交叉熵，只是数据换成指令回答格式。

temperature 越高，分布越平，输出更随机；temperature 越低，分布越尖，输出更保守。

top-k 限制只从概率最高的 `k` 个 token 中采样；top-p 选择累计概率达到 `p` 的最小 token 集合。

KV Cache 用空间换时间，减少 decode 阶段重复计算历史 K/V 的成本。

LoRA 用低秩增量更新大矩阵，大幅降低微调参数量和显存需求。

偏好优化中的 KL 直觉是：提升人类偏好的同时，不让模型语言能力和行为分布漂移太远。

量化常用于推理部署和低显存微调。它通常牺牲少量精度，换取更低显存占用和更高吞吐。

## 面试陷阱

- 自回归不是并行生成所有 token，而是逐 token 条件生成。
- 训练时可以并行计算所有位置的 loss，推理生成时通常必须一步步 decode。
- temperature 不是越高越好，高了会更随机甚至胡说。
- top-k 和 top-p 都是采样截断策略，但一个固定个数，一个固定累计概率。
- KV Cache 减少的是重复计算，不是让 attention 的历史长度消失。
- LoRA 不是压缩原权重，而是在冻结原权重旁边学习低秩增量。

## 自测题

1. 写出自回归语言模型的概率分解。
2. LLM 预训练 loss 为什么可以看成大量 token 分类交叉熵？
3. temperature 变大时，softmax 分布会怎样？
4. top-k 和 top-p 的区别是什么？
5. Attention 中为什么要除以 `sqrt(D_h)`？
6. `Q.shape = (B, H, T, D_h)`，`K.shape = (B, H, T, D_h)`，`QK^T` 的 shape 是什么？
7. KV Cache 在 decode 阶段缓存了什么？
8. LoRA 的低秩增量为什么能减少训练参数量？
9. KL penalty 在偏好优化里防止什么问题？
10. 量化为什么能省显存？它可能带来什么代价？
