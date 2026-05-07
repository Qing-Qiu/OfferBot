# 05 LLM：大模型机制与应用

本章按“机制 -> 训练 -> 推理 -> 应用”组织，不按论文名堆列表。目标是让你能讲清：

```text
Transformer 为什么能并行
Attention 的 shape 和复杂度
LLM 如何训练和生成
RAG 如何把外部知识接入模型
```

## 当前内容

- [01_transformer_basics](./01_transformer_basics/README.md)：token、embedding、position、MHA、FFN、residual、LayerNorm、mask。
- [02_attention_deep_dive](./02_attention_deep_dive/README.md)：Q/K/V、score shape、mask、MHA/MQA/GQA、复杂度、FlashAttention 直觉。
- [03_position_encoding](./03_position_encoding/README.md)：绝对位置编码、sinusoidal、relative bias、RoPE、ALiBi、长上下文外推。
- [06_rag](./06_rag/README.md)：文档解析、切分、embedding、向量库、query rewrite、rerank、评估、幻觉。

## 学习顺序

1. 先学 Transformer 基础，建立 block 结构图。
2. 再深挖 Attention，吃透 Q/K/V 和复杂度。
3. 接着单独补位置编码，理解顺序、相对距离和 RoPE。
4. 最后学 RAG，把模型能力和工程系统接起来。
