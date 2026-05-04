# 07 Model Evaluation：模型评估

## 核心直觉

评估的目标不是证明模型在训练集上多会背题，而是估计它面对新数据时表现如何。

机器学习面试里，评估指标通常考两件事：

```text
指标本身怎么算
这个业务场景为什么该用它
```

回归任务看误差大小，分类任务看判对比例、正负样本排序、阈值后的 precision/recall，推荐任务还要看列表位置。

## 关键概念

训练集：用于更新参数。

验证集：用于调参、选模型、早停。

测试集：用于最终估计泛化能力。

混淆矩阵：把预测和真实标签拆成 TP、FP、TN、FN。

ROC-AUC：看所有阈值下 TPR 和 FPR 的关系，也可以理解为正样本排在负样本前面的概率。

PR-AUC：看 precision 和 recall 的权衡，在正样本稀少时更敏感。

类别不平衡：不同类别样本量差距大，accuracy 容易误导。

## 必要公式 / shape / 流程

回归指标：

```text
MSE = mean((y_hat - y)^2)
RMSE = sqrt(MSE)
MAE = mean(|y_hat - y|)
R2 = 1 - SS_res / SS_tot
```

混淆矩阵：

```text
TP: 真实正，预测正
FP: 真实负，预测正
TN: 真实负，预测负
FN: 真实正，预测负
```

分类指标：

```text
Accuracy = (TP + TN) / (TP + FP + TN + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * Precision * Recall / (Precision + Recall)
```

ROC：

```text
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
```

AUC 概率解释：

```text
随机抽一个正样本和一个负样本
模型给正样本的分数大于负样本的概率
```

评估流程：

```text
先按时间或随机方式切分数据
训练集训练
验证集选超参
测试集只做最终报告
按业务目标选择主指标和辅助指标
```

## 代码阅读提示

看评估代码时，先分清：

```text
score/probability: 连续分数，用于 AUC、PR-AUC
label prediction: 阈值后的 0/1 类别，用于 precision/recall/F1
```

如果代码先把概率阈值化再算 AUC，通常就是错的；AUC 应该用连续分数。

时间序列、推荐、广告等任务常用时间切分，因为随机切分可能让未来信息泄漏到训练集。

## 面试高频问法

1. 训练集、验证集、测试集分别做什么？
2. accuracy 在类别不平衡时为什么可能失效？
3. precision 和 recall 的区别是什么？
4. ROC-AUC 的概率解释是什么？
5. PR-AUC 什么时候比 ROC-AUC 更有参考价值？
6. 回归任务里 MSE、RMSE、MAE 怎么选？
7. 什么是数据泄漏？
8. 离线指标上涨，线上不涨可能有哪些原因？

## 常见陷阱

- 用测试集反复调参，导致测试集被污染。
- 类别不平衡时只报 accuracy。
- 把概率阈值化之后再算 AUC。
- 随机切分有时间依赖的数据，造成时间穿越。
- 只看单个指标，不看业务约束和错误成本。

## 自测题

1. 为什么需要验证集？
2. precision 高但 recall 低，通常意味着什么？
3. recall 高但 precision 低，通常意味着什么？
4. AUC 的概率解释是什么？
5. 正样本极少时，PR-AUC 为什么常比 ROC-AUC 更敏感？
6. MSE 和 MAE 哪个更受异常值影响？
7. 什么是数据泄漏？举一个例子。
8. 为什么推荐系统常用时间切分？
