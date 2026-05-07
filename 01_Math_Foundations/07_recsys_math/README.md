# 07 RecSys Math：推荐系统数学

## 核心直觉

推荐系统数学主要解决三件事：

```text
怎么表示用户和物品
怎么计算匹配分数
怎么评估排序结果好不好
```

召回阶段更关心相似度和检索效率，排序阶段更关心概率、loss、特征交叉和校准。

## 必要公式

点积匹配：

$$
\operatorname{score}(u,v)=u\cdot v
$$

余弦相似度：

$$
\cos(u,v)=\frac{u\cdot v}{\|u\|\,\|v\|}
$$

矩阵分解：

$$
\hat{R}=UV^\top,\qquad \hat{r}_{ij}=u_i\cdot v_j
$$

FM 二阶交叉：

$$
\hat{y}=w_0+\sum_i w_ix_i+\sum_{i<j}\langle v_i,v_j\rangle x_ix_j
$$

FM 二阶项高效计算：

$$
\frac{1}{2}\sum_f\left[\left(\sum_i v_{if}x_i\right)^2-\sum_i v_{if}^2x_i^2\right]
$$

BPR loss：

$$
L=-\log\sigma\left(s(u,i^+)-s(u,i^-)\right)
$$

AUC 直觉：

```text
随机抽一个正样本和一个负样本
模型把正样本排在负样本前面的概率
```

DCG / NDCG：

$$
\operatorname{DCG}@K=\sum_{i=1}^{K}\frac{\operatorname{rel}_i}{\log_2(i+1)},\qquad
\operatorname{NDCG}@K=\frac{\operatorname{DCG}@K}{\operatorname{IDCG}@K}
$$

MAP 直觉：

```text
对每个用户计算 Average Precision
再对所有用户取平均
```

多目标加权：

$$
\operatorname{score}=w_1\operatorname{CTR}+w_2\operatorname{CVR}+w_3\operatorname{stay\_time}+\cdots
$$

校准：

```text
预测概率 0.8 的样本中，真实发生比例也应接近 0.8
```

位置偏差：

```text
用户更容易点击靠前位置
点击不完全等于兴趣
```

## AI 用法

双塔模型用用户塔和物品塔分别生成向量，再用点积或余弦相似度做召回。

矩阵分解是 embedding 推荐的经典起点：用户和物品都被表示成低维向量，交互分数来自向量内积。

DeepFM、Wide&Deep 等模型关注特征交叉。FM 的核心是用隐向量建模二阶组合，避免为每个交叉特征单独学参数。

BPR 适合隐式反馈排序，因为很多场景只有点击、观看、购买等正反馈，没有可靠的显式负反馈。

AUC、NDCG、MAP 用于评估排序质量。AUC 更偏 pairwise，NDCG 更关注头部位置和相关性强弱。

多目标排序需要在点击、转化、时长、满意度等目标之间权衡。位置偏差建模则用于区分“因为用户喜欢而点击”和“因为排在前面而点击”。

## 面试陷阱

- 点积受向量长度影响，余弦相似度更关注方向。
- 负采样会改变训练分布，线上预估概率可能需要校准。
- AUC 不关心绝对分数，只关心正负样本相对顺序。
- NDCG 对靠前位置更敏感，适合评估推荐列表头部质量。
- 矩阵分解处理冷启动能力弱，需要内容特征或 side information。
- 召回指标和排序指标不同，不能只用一个指标概括整个推荐链路。

## 自测题

1. 双塔召回为什么常用点积或余弦相似度？
2. 点积和余弦相似度有什么区别？
3. 矩阵分解里 `R_hat = U V^T` 的每一项表示什么？
4. FM 为什么能建模二阶特征交叉？
5. BPR loss 想让正样本和负样本的分数满足什么关系？
6. AUC 的概率解释是什么？
7. NDCG 为什么比普通准确率更适合推荐列表？
8. 负采样会带来什么训练和校准问题？
9. 为什么矩阵分解容易遇到冷启动问题？
10. 多目标排序为什么不能简单只优化 CTR？
11. 位置偏差会如何污染点击标签？
