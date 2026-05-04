# 04 Loss Functions：损失函数

## 核心直觉

loss 是训练时的方向盘。模型预测错了多少、往哪个方向改参数，都由 loss 和它的梯度决定。

metric 是评估时的成绩单。业务真正关心的可能是 AUC、Recall、NDCG、GMV，但这些指标不一定可导、不一定适合直接优化，所以训练时常用更平滑、更稳定的 surrogate loss。

一句话：

```text
loss 用来训练，metric 用来评估。
```

## 关键概念

`MSE`：均方误差，回归任务常用，对异常值敏感。

`MAE`：平均绝对误差，回归任务常用，比 MSE 更抗异常值，但梯度不如 MSE 平滑。

`BCE`：二分类交叉熵，优化真实二分类标签的 log likelihood。

`Cross Entropy`：多分类交叉熵，惩罚真实类别预测概率太低。

`LogLoss`：二分类场景中常指 BCE 的平均形式。

`surrogate loss`：替代损失。真实业务指标难优化时，用可导 loss 替代。

## 必要公式 / shape / 流程

MSE：

```text
L = (1/n) sum_i (y_hat_i - y_i)^2
dL/dy_hat = (2/n)(y_hat - y)
```

MAE：

```text
L = (1/n) sum_i |y_hat_i - y_i|
```

BCE：

```text
L = -(1/n) sum_i [y_i log p_i + (1-y_i) log(1-p_i)]
```

多分类交叉熵：

```text
logits.shape = (n, C)
target.shape = (n,)
p = softmax(logits)
L = -(1/n) sum_i log p_i,target_i
```

softmax + cross entropy 梯度：

```text
dL/dlogits = p - y_one_hot
```

loss 选择流程：

```text
回归 -> MSE / MAE / Huber
二分类 -> BCE / BCEWithLogits
多分类单标签 -> CrossEntropy
多标签分类 -> 多个 sigmoid + BCE
排序 -> pairwise / listwise loss
```

## 代码阅读提示

先看模型输出是什么：

```text
输出连续值 -> 回归 loss
输出一个 logit -> BCEWithLogitsLoss
输出 C 个 logits -> CrossEntropyLoss
输出一组排序分数 -> BPR / pairwise / listwise
```

再看标签 shape：

```text
二分类标签通常是 0/1
多分类标签通常是类别 index
多标签标签通常是 multi-hot 向量
```

最常见的 PyTorch 读代码陷阱是：`CrossEntropyLoss` 的输入是 logits，标签是类别编号，不是 one-hot，也不是 softmax 后的概率。

## 面试高频问法

1. loss 和 metric 的区别是什么？
2. 为什么回归常用 MSE？
3. MSE 为什么对异常值敏感？
4. 二分类为什么常用 BCE？
5. 多分类交叉熵为什么等价于最大似然？
6. `CrossEntropyLoss` 和 `NLLLoss` 有什么关系？
7. 类别不平衡时 loss 可以怎么改？
8. 为什么 AUC 不常直接作为训练 loss？

## 常见陷阱

- 只看 loss，不看业务 metric。loss 降低不一定代表业务指标提升。
- 把多分类和多标签混淆。多分类是互斥类别，多标签可以多个类别同时为真。
- 对 logits 手动 softmax 后再传给 `CrossEntropyLoss`。
- 用 accuracy 评估极度类别不平衡任务。
- MSE 在异常值存在时可能被少数大误差主导。

## 自测题

1. loss 和 metric 各自用于什么阶段？
2. MSE 和 MAE 对异常值的敏感程度有什么区别？
3. 写出二分类 BCE。
4. 为什么交叉熵适合分类任务？
5. `logits.shape = (32, 10)`，`target.shape` 在 PyTorch 多分类里通常是什么？
6. 多标签分类应该用 softmax 还是多个 sigmoid？
7. 为什么 AUC 不容易直接优化？
8. 类别不平衡时，可以从采样、loss 权重、阈值三方面怎么处理？
