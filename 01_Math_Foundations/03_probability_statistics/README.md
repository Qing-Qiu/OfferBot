# 03 Probability Statistics：概率统计基础

## 核心直觉

概率是给不确定性建模，统计是从数据里估计规律。机器学习里的模型通常不是直接说“答案一定是什么”，而是学习：

```text
给定输入 x，标签 y 出现的概率有多大
```

训练模型时，我们希望真实数据在模型下出现的概率尽可能大。这就是最大似然估计的核心。

## 必要公式

条件概率：

```text
P(A | B) = P(A, B) / P(B)
```

贝叶斯公式：

```text
P(A | B) = P(B | A) P(A) / P(B)
```

期望：

```text
E[X] = sum_x x P(X = x)
```

方差：

```text
Var(X) = E[(X - E[X])^2]
```

协方差：

```text
Cov(X, Y) = E[(X - E[X])(Y - E[Y])]
```

常见分布：

```text
Bernoulli: 单次 0/1 事件，例如是否点击
Binomial: 多次 Bernoulli 成功次数
Gaussian: 连续变量的钟形分布
Categorical: 多类别离散分布，例如 token 类别
```

最大似然估计 MLE：

```text
theta_hat = argmax_theta product_i P(y_i | x_i; theta)
```

实践中常取 log：

```text
theta_hat = argmax_theta sum_i log P(y_i | x_i; theta)
```

最大后验估计 MAP：

```text
theta_hat = argmax_theta P(theta | D)
= argmax_theta P(D | theta) P(theta)
```

经验风险最小化：

```text
min_theta (1/n) sum_i L(f_theta(x_i), y_i)
```

## AI 用法

分类模型输出的 softmax 可以看成 `P(y | x)`。用交叉熵训练分类器，本质上等价于最大化真实标签的 log likelihood。

正则化可以从 MAP 角度理解：给参数加先验。比如 L2 正则相当于偏好较小的参数。

推荐系统里的点击、转化、曝光都可以看成随机事件。CTR 模型学习的是：

```text
P(click = 1 | user, item, context)
```

A/B 实验用统计方法判断两个策略的指标差异是真效果，还是随机波动。

采样和随机种子用于让训练、实验和负采样过程可复现。数据分布漂移则提醒我们：训练分布和线上分布不一致时，离线效果可能失真。

## 面试陷阱

- 概率 `P(data | theta)` 和后验 `P(theta | data)` 不是一回事。
- likelihood 是把数据固定、参数当变量；普通概率常把参数固定、事件当变量。
- 相关不等于因果。
- p-value 不是“原假设为真的概率”，而是在原假设成立时看到当前或更极端结果的概率。
- 训练集 loss 低不代表泛化好，统计学习关注的是未见数据上的风险。

## 自测题

1. 用一句话解释条件概率 `P(A | B)`。
2. 写出贝叶斯公式，并说明先验、似然、后验分别是什么。
3. 期望和方差分别刻画什么？
4. 为什么训练分类模型常用 log likelihood，而不是直接乘很多概率？
5. MLE 和 MAP 的区别是什么？
6. L2 正则可以从什么先验直觉理解？
7. 过拟合和泛化误差是什么关系？
8. p-value 能不能解释成“原假设为真的概率”？为什么？
9. CTR 模型中的 label 通常是什么随机变量？
10. A/B 实验为什么需要样本量，而不能只看一天指标涨跌？
11. Bernoulli 分布和 Categorical 分布分别适合描述什么任务？
12. 为什么训练集和线上数据分布漂移会伤害模型效果？
