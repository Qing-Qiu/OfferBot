# 04 Ranking Models：排序模型

## 核心直觉

排序层面对的是召回层给出的候选集。候选数量已经从百万级降到几百或几千，所以排序模型可以更复杂，使用更多特征，目标是更精细地估计：

```text
用户在当前上下文下，对这个物品会不会点击、转化、停留、满意
```

排序模型的发展主线是：从线性模型到特征交叉，再到深度网络和用户行为序列建模。

## 关键概念

`LR`：逻辑回归，最经典 CTR 预估基线，依赖人工特征交叉。

`GBDT + LR`：GBDT 自动生成高阶组合特征，再喂给 LR。

`Wide & Deep`：Wide 记忆强规则，Deep 学泛化表示。

`FM`：用隐向量建模二阶特征交叉。

`DeepFM`：FM 负责低阶交叉，DNN 负责高阶非线性。

`DCN`：显式建模 bounded-degree feature crossing。

`DIN`：根据候选 item 对用户历史行为做 attention，建模兴趣相关性。

`MMOE / PLE`：多任务学习模型，用于 CTR、CVR 等多目标。

## 必要公式 / shape / 流程

LR：

```text
logit = w · x + b
p = sigmoid(logit)
```

FM：

```text
y_hat = w0 + sum_i w_i x_i + sum_{i<j} <v_i, v_j> x_i x_j
```

DeepFM：

```text
output = sigmoid(y_FM + y_DNN)
```

DIN attention 直觉：

```text
candidate item 作为 query
用户历史行为作为 key/value
对和候选更相关的历史行为赋更大权重
```

Pointwise BCE：

```text
L = -[y log p + (1-y) log(1-p)]
```

Pairwise BPR：

```text
L = -log sigmoid(score_pos - score_neg)
```

排序模型输入常见结构：

```text
user features
item features
context features
user behavior sequence
cross features
```

## 代码阅读提示

读排序模型代码时先看特征：

```text
sparse feature -> embedding
dense feature -> normalize / concat
sequence feature -> pooling / attention
cross feature -> FM / DCN / manual cross
```

再看模型输出：

```text
CTR/CVR -> 一个 logit
多任务 -> 多个 head
排序分数 -> 可能是概率，也可能是融合分
```

DeepFM 类代码里，重点区分 FM 分支和 DNN 分支是否共享 embedding。DIN 类代码里，重点看 attention 的 query 是否来自候选 item。

## 面试高频问法

1. LR 为什么在推荐系统里很重要？
2. Wide & Deep 的 Wide 和 Deep 各自解决什么？
3. FM 如何建模二阶特征交叉？
4. DeepFM 相比 FM 多了什么？
5. DCN 和 FM 的特征交叉有什么区别？
6. DIN 为什么要对用户历史行为做 attention？
7. Pointwise、Pairwise、Listwise loss 有什么区别？
8. 多任务排序为什么会用 MMOE / PLE？

## 常见陷阱

- 只背模型名，不讲它解决的链路问题。
- 忽略特征工程，推荐排序模型很大一部分效果来自特征质量。
- 把双塔召回模型和精排深度交叉模型混为一谈。
- 只优化 CTR，可能导致标题党、低转化或低满意度。
- 序列模型里不注意时间顺序，容易产生未来信息泄漏。
- 多任务学习里任务之间可能冲突，不是 head 越多越好。

## 自测题

1. 为什么 LR 仍然是推荐系统重要基线？
2. FM 的二阶交叉公式是什么？
3. DeepFM 的 FM 分支和 DNN 分支分别学什么？
4. DIN 中 candidate item 为什么可以作为 attention query？
5. Pointwise 和 Pairwise loss 的训练信号有什么区别？
6. 排序模型常见的 user/item/context 特征各举两个例子。
7. 为什么排序模型通常比召回模型复杂？
8. 多目标排序可能出现哪些目标冲突？
