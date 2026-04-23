# 01 数学基础知识树

目标：服务机器学习、深度学习、推荐系统、大模型；重点学“能看懂公式、能解释直觉、能推关键梯度、能分析复杂度”。

## 建议小章节

```text
01_linear_algebra
02_calculus_and_gradient
03_probability
04_statistics
05_information_theory
06_optimization
07_matrix_derivatives
08_ai_formula_playbook
```

说明：上面的小章节对应学习模块；下面的 A/B/C... 继续保留为可展开的叶子知识点树。

## A. 线性代数

- A1 标量、向量、矩阵、张量
- A2 向量加法、数乘、点积
- A3 矩阵乘法
- A4 转置、单位矩阵、逆矩阵
- A5 线性变换的几何意义
- A6 向量范数、矩阵范数
- A7 正交、投影、余弦相似度
- A8 秩、线性相关、线性无关
- A9 特征值、特征向量
- A10 SVD 奇异值分解
- A11 PCA 主成分分析
- A12 Embedding 向量空间直觉

## B. 微积分与梯度

- B1 函数、极限、连续
- B2 导数的几何意义
- B3 常见函数求导
- B4 偏导数
- B5 梯度
- B6 链式法则
- B7 Jacobian
- B8 Hessian
- B9 泰勒展开
- B10 梯度下降的数学解释
- B11 反向传播中的链式法则
- B12 梯度消失、梯度爆炸的数学直觉

## C. 概率论

- C1 随机变量
- C2 概率分布
- C3 条件概率
- C4 贝叶斯公式
- C5 期望、方差、协方差
- C6 常见分布：Bernoulli、Binomial、Gaussian、Categorical
- C7 最大似然估计 MLE
- C8 最大后验估计 MAP
- C9 独立性、条件独立
- C10 采样、随机性、seed
- C11 蒙特卡洛估计
- C12 推荐系统中的曝光/点击概率

## D. 统计学习基础

- D1 估计量、偏差、方差
- D2 bias-variance tradeoff
- D3 经验风险最小化 ERM
- D4 泛化误差
- D5 过拟合、欠拟合
- D6 正则化的统计意义
- D7 置信区间
- D8 假设检验
- D9 p-value 基本直觉
- D10 A/B 实验统计基础
- D11 长尾分布
- D12 数据分布漂移

## E. 信息论

- E1 信息量
- E2 熵 Entropy
- E3 交叉熵 Cross Entropy
- E4 KL 散度
- E5 JS 散度
- E6 互信息
- E7 困惑度 Perplexity
- E8 为什么分类常用交叉熵
- E9 KL 在蒸馏、RLHF、VAE 中的作用

## F. 优化理论基础

- F1 目标函数
- F2 凸函数、非凸函数
- F3 局部最优、全局最优、鞍点
- F4 学习率
- F5 梯度下降
- F6 SGD
- F7 Momentum
- F8 Adam / AdamW 直觉
- F9 L1 / L2 正则
- F10 约束优化
- F11 拉格朗日乘子
- F12 数值稳定性：overflow、underflow、log-sum-exp

## G. 矩阵求导与深度学习常用公式

- G1 `y = Xw + b` 的梯度
- G2 MSE 的梯度
- G3 sigmoid 的导数
- G4 BCE 的梯度
- G5 softmax 的导数
- G6 cross entropy + softmax 的梯度
- G7 LayerNorm / BatchNorm 的核心统计量
- G8 attention score 的矩阵维度
- G9 `QK^T / sqrt(d)` 的 shape 推导
- G10 embedding lookup 的梯度直觉

## H. 推荐系统数学

- H1 相似度：cosine、dot product、Euclidean
- H2 矩阵分解
- H3 FM 二阶交叉公式
- H4 BPR loss
- H5 AUC 数学含义
- H6 NDCG / MAP
- H7 负采样概率
- H8 多目标加权
- H9 校准 calibration
- H10 位置偏差建模

## I. 大模型数学

- I1 token 概率建模
- I2 自回归分解
- I3 softmax 采样
- I4 temperature 的数学作用
- I5 top-k / top-p
- I6 attention 矩阵计算
- I7 KV Cache 的复杂度变化
- I8 LoRA 低秩分解
- I9 量化误差
- I10 KL penalty 与偏好优化直觉

## 优先级

```text
必学：
A1-A7
B2-B6, B10-B12
C1-C8
D1-D6
E2-E4, E7-E8
F1-F9, F12
G1-G6, G8-G10
H1-H6
I1-I8

次学：
A8-A12
B7-B9
C9-C12
D7-D12
E1, E5-E6, E9
F10-F11
G7
H7-H10
I9-I10
```
