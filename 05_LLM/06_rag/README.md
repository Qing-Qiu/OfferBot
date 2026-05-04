# 06 RAG：检索增强生成

## 核心直觉

RAG 的核心想法是：不要指望模型参数记住所有知识，而是在回答前先从外部知识库检索相关资料，再把资料放进上下文让模型生成答案。

典型流程：

```text
用户问题
-> query rewrite
-> 检索相关文档 chunk
-> rerank
-> 组装 prompt
-> LLM 生成答案
-> 引用与评估
```

RAG 不是“接个向量库”这么简单。真正难的是文档质量、切分粒度、召回质量、上下文压缩、权限隔离和评估。

## 关键概念

`文档解析`：从 PDF、网页、Word、表格等来源提取文本和结构。

`切分 chunking`：把长文档切成可检索片段。

`embedding`：把文本映射成向量，用于语义检索。

`向量库`：存储 embedding 并支持 ANN 检索。

`query rewrite`：改写用户问题，让它更适合检索。

`HyDE`：先让模型生成假想答案，再用假想答案做检索。

`多路召回`：结合向量检索、关键词检索、结构化过滤等。

`rerank`：对召回结果做更精细的相关性排序。

`上下文压缩`：把候选资料压成更少、更相关的上下文。

`忠实性`：答案是否被检索材料支持。

## 必要公式 / shape / 流程

向量检索：

```text
query_emb.shape = (D,)
doc_embs.shape = (N, D)
scores = doc_embs @ query_emb
topK = argsort(scores)
```

RAG 数据链路：

```text
load documents
-> parse
-> clean
-> chunk
-> embed
-> index
```

RAG 查询链路：

```text
question
-> rewrite / expand
-> retrieve topK
-> rerank topM
-> build context
-> generate answer with citations
```

评估维度：

```text
retrieval recall: 正确证据有没有被召回
faithfulness: 回答是否忠于证据
relevance: 回答是否回答了问题
latency: 检索和生成是否足够快
```

## 代码阅读提示

读 RAG 代码时先拆成两条链：

```text
离线入库链：parse -> chunk -> embed -> upsert index
在线问答链：rewrite -> retrieve -> rerank -> prompt -> generate
```

再看关键参数：

```text
chunk size / overlap
embedding model
topK / topM
similarity metric
reranker
prompt template
max context length
```

如果效果差，不要只怪 LLM。很多 RAG 问题来自检索阶段没有找到正确证据。

## 面试高频问法

1. RAG 如何缓解幻觉？
2. RAG 和微调有什么区别？
3. chunk size 太大或太小分别有什么问题？
4. 向量召回和关键词召回各自优缺点是什么？
5. rerank 的作用是什么？
6. 如何评估 RAG 系统？
7. RAG 为什么仍然会幻觉？
8. 企业知识库 RAG 如何处理权限？

## 常见陷阱

- 只做向量召回，不处理文档解析和 chunk 质量。
- topK 过小导致漏证据，topK 过大导致上下文噪声多。
- 没有 rerank，直接把粗召回结果塞给模型。
- 没有引用和证据约束，生成答案不可追溯。
- 忽略权限过滤，导致越权检索敏感文档。
- 只看生成质量，不单独评估检索质量。

## 自测题

1. RAG 的离线链路和在线链路分别是什么？
2. RAG 为什么能缓解但不能彻底消灭幻觉？
3. chunk size 太大会怎样？太小会怎样？
4. query rewrite 解决什么问题？
5. rerank 为什么通常比初召回更精细？
6. 如何判断 RAG 的问题出在检索还是生成？
7. 企业 RAG 为什么必须做权限过滤？
8. RAG 和 fine-tuning 分别适合解决什么问题？
