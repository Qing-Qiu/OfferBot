# 01 Math Foundations：AI 数学基础

本章服务机器学习、深度学习、推荐系统和大模型，不做纯数学竞赛式展开。目标不是把数学书重写一遍，而是让你在面试和代码里能稳定做到：

```text
能看懂公式
能解释直觉
能推关键梯度
能分析复杂度和工程影响
```

## 学习策略

你的线性代数和微积分基础已经有底子，所以本章采用“薄弱优先”的路线：

```text
线性代数：只做 AI 高频复习
链式法则：重点加强到反向传播
概率统计：夯实 MLE / MAP / 泛化 / A/B
信息论：夯实交叉熵 / KL / 困惑度
优化：夯实 SGD / AdamW / 正则化 / 数值稳定性
矩阵求导：重点补
推荐数学：重点补
大模型数学：重点补
```

## 小章节

1. [01_linear_algebra_review](./01_linear_algebra_review/README.md)：shape、矩阵乘法、相似度、PCA/SVD、Attention shape。
2. [02_chain_rule_and_gradient](./02_chain_rule_and_gradient/README.md)：导数、偏导、梯度、链式法则、反向传播直觉。
3. [03_probability_statistics](./03_probability_statistics/README.md)：条件概率、贝叶斯、期望方差、MLE/MAP、泛化、A/B。
4. [04_information_theory](./04_information_theory/README.md)：熵、交叉熵、KL、困惑度、分类 loss 与 LLM loss。
5. [05_optimization](./05_optimization/README.md)：目标函数、SGD、Momentum、Adam/AdamW、正则化、数值稳定性。
6. [06_matrix_derivatives](./06_matrix_derivatives/README.md)：`Xw+b`、MSE、sigmoid、BCE、softmax、cross entropy、矩阵反传。
7. [07_recsys_math](./07_recsys_math/README.md)：相似度、矩阵分解、FM、BPR、AUC、NDCG、负采样、校准。
8. [08_llm_math](./08_llm_math/README.md)：自回归概率、sampling、Attention、KV Cache、LoRA、KL penalty。
9. [09_formula_playbook](./09_formula_playbook/README.md)：AI 面试高频公式速查表。

## 统一符号

```text
n: 样本数
d: 特征维度
B: batch size
T: 序列长度
D: embedding / hidden 维度
C: 类别数
X: 特征矩阵
w, W: 参数向量 / 参数矩阵
b: 偏置
y: 标签
y_hat: 预测值
L: loss
```

## 每节固定结构

```text
核心直觉
必要公式
AI 用法
面试陷阱
自测题
```
