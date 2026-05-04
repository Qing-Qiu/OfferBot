# 02 Recall：召回

## 核心直觉

召回的任务是在很短时间内从海量物品里找出一批“可能相关”的候选。它不追求最终排序最精细，而追求：

```text
快
覆盖足够广
不要漏掉真正可能感兴趣的物品
```

召回通常是多路的：热门召回、规则召回、协同过滤、向量召回、兴趣标签召回等一起工作，再做融合和去重。

## 关键概念

`UserCF`：找相似用户，把相似用户喜欢的物品推荐给当前用户。

`ItemCF`：找相似物品，根据用户最近互动过的物品扩展候选。

`Swing`：强调用户共同点击物品时，越“小众”的共同行为越有信息量，常用于电商相似物品。

`矩阵分解`：把用户和物品映射到低维向量，用内积表示偏好。

`向量召回`：用模型生成 user embedding 和 item embedding，再做 ANN 检索。

`ANN`：approximate nearest neighbor，近似最近邻。用可接受的近似误差换检索速度。

`召回融合`：合并多路候选，做去重、截断、配额和权重控制。

## 必要公式 / shape / 流程

UserCF 相似度：

```text
sim(u, v) = |I_u ∩ I_v| / sqrt(|I_u| |I_v|)
```

ItemCF 相似度：

```text
sim(i, j) = |U_i ∩ U_j| / sqrt(|U_i| |U_j|)
```

矩阵分解：

```text
R_hat = U V^T
score(u, i) = u_emb · item_emb
```

双塔向量召回：

```text
user_emb.shape = (B, D)
item_emb.shape = (N, D)
scores = user_emb @ item_emb.T
```

ANN 流程：

```text
离线生成 item embedding
构建向量索引
在线生成 user/query embedding
检索 topK item
返回候选给排序层
```

召回评估：

```text
Recall@K = 命中的正样本数 / 全部正样本数
HitRate@K = 是否至少命中一个正样本
Coverage = 被推荐过的物品覆盖率
```

## 代码阅读提示

读召回代码先看候选来源：

```text
基于行为共现 -> CF / Swing
基于 embedding -> 双塔 / 向量检索
基于规则 -> 热门 / 类目 / 地理位置
基于内容 -> 标签 / 文本 / 图片 embedding
```

再看在线代价：是否需要全量打分，是否用了索引，索引多久更新，item embedding 是否离线预计算。

向量召回代码里，最关键的是 shape 和归一化：

```text
dot product: 受向量长度影响
cosine: 通常先 normalize，再做内积
```

## 面试高频问法

1. 召回和排序的区别是什么？
2. UserCF 和 ItemCF 各自适合什么场景？
3. Swing 相比 ItemCF 想解决什么问题？
4. 双塔为什么适合召回？
5. 向量召回为什么需要 ANN？
6. Faiss / HNSW / IVF / PQ 的核心直觉是什么？
7. 召回怎么评估？
8. 多路召回怎么融合？

## 常见陷阱

- 用精排思维回答召回，忽略候选规模和延迟。
- 只看 Recall@K，不看覆盖率、多样性和后续排序质量。
- 双塔把用户和物品分开编码，线上快，但表达能力弱于深度交叉排序模型。
- 负采样策略会显著影响向量空间质量。
- ANN 是近似检索，召回率和延迟之间需要权衡。
- 索引不更新会导致新物品召不出来。

## 自测题

1. 为什么召回层通常是多路召回？
2. UserCF 和 ItemCF 的相似度分别基于什么集合？
3. 双塔线上为什么可以提前构建 item 索引？
4. `user_emb.shape=(B,D)`，`item_emb.shape=(N,D)`，全量打分 shape 是什么？
5. ANN 为什么不是精确最近邻？
6. Recall@K 和 HitRate@K 有什么区别？
7. 双塔召回的表达能力有什么局限？
8. 新物品冷启动召回可以怎么做？
