# 03 Logistic Regression：逻辑回归

## 核心直觉

逻辑回归是最重要的分类基线模型。它的名字里有“回归”，但它通常用于分类，因为它先做一个线性打分，再把分数压成概率：

```text
线性打分：z = x · w + b
概率输出：p = sigmoid(z)
```

`z` 叫 logit。logit 不受 `0` 到 `1` 限制，可以是任意实数；`sigmoid(z)` 把它映射成正类概率。模型训练的目标是：正样本的 `p` 尽量接近 `1`，负样本的 `p` 尽量接近 `0`。

逻辑回归非常适合建立分类任务的第一把尺子：能解释、训练快、容易排查数据和特征问题。推荐系统 CTR 预估、广告点击率预估、风控二分类都能看到它的影子。

## 关键概念

`logits`：模型在概率化之前的原始分数。二分类里是一个实数，多分类里是一组类别分数。

`sigmoid`：把任意实数映射到 `(0, 1)`，常用于二分类概率。

`BCE`：binary cross entropy，二分类交叉熵。它惩罚“真实类别概率太低”的情况。

`decision threshold`：决策阈值。默认常用 `0.5`，但业务上可以根据 precision / recall 成本调整。

`softmax regression`：多分类版逻辑回归。输出每个类别的 logit，再用 softmax 得到类别概率分布。

## 必要公式 / shape / 流程

二分类输入：

```text
X.shape = (n, d)
w.shape = (d, 1)
b.shape = 标量
y.shape = (n, 1), y in {0, 1}
```

前向传播：

```text
z = Xw + b
p = sigmoid(z) = 1 / (1 + exp(-z))
```

BCE loss：

```text
L = -(1/n) sum_i [y_i log p_i + (1-y_i) log(1-p_i)]
```

sigmoid + BCE 的关键梯度：

```text
dL/dz = p - y
dL/dw = (1/n) X^T (p - y)
dL/db = mean(p - y)
```

多分类 softmax：

```text
logits.shape = (n, C)
p_i = exp(z_i) / sum_j exp(z_j)
CrossEntropy = -log p_true_class
```

## 代码阅读提示

看到逻辑回归代码时，先找三件事：

```text
1. logits 怎么算：通常是 X @ w + b
2. loss 是否直接接收 logits：稳定实现通常不先手动 sigmoid/softmax
3. 阈值在哪里：训练用概率，决策才用 threshold
```

在 PyTorch 里，二分类更推荐 `BCEWithLogitsLoss`，它把 sigmoid 和 BCE 合在一起做数值稳定计算。多分类更推荐 `CrossEntropyLoss`，它接收 raw logits，不需要你先 softmax。

## 面试高频问法

1. 逻辑回归为什么能做分类？
2. 逻辑回归为什么叫“回归”？
3. sigmoid 的作用是什么？
4. logits 和 probability 的区别是什么？
5. 为什么二分类常用 BCE，不用 MSE？
6. 阈值一定要取 `0.5` 吗？
7. 多分类逻辑回归和二分类逻辑回归有什么区别？
8. `BCEWithLogitsLoss` 为什么比手写 `sigmoid + BCE` 更稳定？

## 常见陷阱

- 把 logit 当概率。logit 可以大于 `1` 或小于 `0`，概率才在 `[0, 1]`。
- 训练阶段和决策阶段混在一起。训练优化的是概率分布，阈值只用于最终分类。
- 类别不平衡时只看 accuracy。极端情况下全部预测多数类也可能 accuracy 很高。
- 手写 BCE 时忘记加 epsilon，`log(0)` 会导致数值问题。
- 多分类时先 softmax 再传给 PyTorch `CrossEntropyLoss`，这会重复处理概率化。

## 自测题

1. 二分类逻辑回归中，`z = x · w + b` 的 `z` 叫什么？
2. sigmoid 的输出范围是什么？
3. 写出 BCE loss。
4. 为什么 `sigmoid + BCE` 的梯度可以简化成 `p - y`？
5. 阈值从 `0.5` 调高到 `0.8`，precision 和 recall 通常会怎样变化？
6. 多分类任务为什么用 softmax，而不是对每个类别单独 sigmoid？
7. PyTorch `CrossEntropyLoss` 为什么接收 logits？
8. 逻辑回归适合做复杂非线性分类边界吗？为什么？
