# 05 大模型知识树

目标：理解 Transformer、训练、推理、RAG、Agent 和应用工程，能讲原理、复杂度、工程权衡。

## 建议小章节

```text
01_transformer_basics
02_attention_deep_dive
03_position_encoding
04_training_and_alignment
05_inference_optimization
06_rag
07_agent_and_tool_calling
08_llm_app_engineering
09_llm_interview
```

说明：大模型部分优先按“机制 -> 训练 -> 推理 -> 应用”展开，不建议一开始就按论文名拆。

## A. Transformer 基础

- A1 Tokenization：BPE / WordPiece / SentencePiece
- A2 Token Embedding
- A3 位置编码：absolute PE / sinusoidal / relative bias / RoPE / ALiBi
- A4 Scaled Dot-Product Attention
- A5 Multi-Head Attention
- A6 FFN
- A7 Residual Connection
- A8 LayerNorm：PreNorm / PostNorm
- A9 Encoder / Decoder
- A10 Causal Mask / Padding Mask

## B. Attention 深入

- B1 Q / K / V 含义
- B2 矩阵维度
- B3 attention score
- B4 softmax
- B5 mask 机制
- B6 MHA / MQA / GQA
- B7 Attention 复杂度
- B8 FlashAttention
- B9 长上下文瓶颈

## C. 预训练与后训练

- C1 自回归建模
- C2 MLM vs CLM
- C3 数据清洗、去重、污染
- C4 SFT
- C5 Reward Model
- C6 RLHF / PPO
- C7 DPO
- C8 LoRA
- C9 QLoRA
- C10 Adapter

## D. 推理优化

- D1 Prefill / Decode
- D2 KV Cache
- D3 Temperature
- D4 Top-k / Top-p
- D5 Beam Search
- D6 量化：INT8 / INT4 / GPTQ / AWQ
- D7 batching / continuous batching
- D8 投机解码
- D9 显存估算
- D10 延迟与吞吐

## E. RAG

- E1 文档解析
- E2 清洗与切分
- E3 Embedding
- E4 向量库与 ANN
- E5 Query Rewrite
- E6 HyDE
- E7 多路召回
- E8 Rerank
- E9 上下文压缩
- E10 引用与答案生成
- E11 RAG 评估：召回率 / 忠实性 / 相关性

## F. Agent 与工具调用

- F1 Tool Calling
- F2 ReAct
- F3 Planner-Executor
- F4 记忆机制
- F5 多 Agent
- F6 任务分解
- F7 错误恢复
- F8 工具权限

## G. 应用工程

- G1 Prompt 结构
- G2 Few-shot
- G3 结构化输出
- G4 JSON Schema
- G5 缓存、重试、限流
- G6 评测集
- G7 日志与 tracing
- G8 Prompt Injection
- G9 敏感信息保护
- G10 质量监控

## H. 面试重点

- H1 Transformer 为什么并行
- H2 Attention 复杂度
- H3 为什么需要位置编码：顺序感、相对距离、RoPE 旋转直觉
- H4 PreNorm vs PostNorm
- H5 KV Cache 节省什么
- H6 RAG 如何缓解幻觉
- H7 微调 vs RAG
- H8 LoRA 为什么参数高效

## 优先级

```text
必学：A, B1-B7, C1-C8, D1-D7, E, G1-G8, H
次学：B8-B9, C9-C10, D8-D10, F, G9-G10
```
