# 02 机器学习基础知识树

目标：掌握参数化模型、损失函数、梯度下降、泛化评估，为深度学习铺地基。

## 建议小章节

```text
01_supervised_learning
02_linear_regression
03_logistic_regression
04_loss_functions
05_gradient_descent
06_regularization
07_model_evaluation
08_feature_basics
```

说明：后续生成目录时以这些小章节为主；下面的 A/B/C... 继续作为知识点展开索引。

## A. 监督学习基本语言

- A1 样本、特征、标签
- A2 训练集、验证集、测试集
- A3 参数 vs 超参数
- A4 模型、假设空间、目标函数
- A5 loss vs metric
- A6 经验风险 vs 泛化误差
- A7 欠拟合、过拟合
- A8 数据泄漏

## B. 线性回归

- B1 模型：`y = Xw + b`
- B2 MSE 损失
- B3 `dL/dw`、`dL/db`
- B4 梯度下降训练
- B5 闭式解直觉
- B6 多特征线性回归
- B7 标准化对训练的影响
- B8 异常值影响
- B9 面试追问：为什么 MSE 对异常值敏感

## C. 逻辑回归

- C1 二分类问题定义
- C2 logits
- C3 sigmoid
- C4 BCE 损失
- C5 决策阈值
- C6 多分类 softmax 回归
- C7 交叉熵
- C8 面试追问：逻辑回归为什么叫回归

## D. 优化基础

- D1 导数、偏导、梯度
- D2 梯度下降
- D3 学习率
- D4 batch / mini-batch / epoch
- D5 随机梯度下降 SGD
- D6 Momentum
- D7 梯度检查
- D8 收敛、震荡、发散

## E. 正则化与模型选择

- E1 L1 正则
- E2 L2 正则
- E3 Elastic Net
- E4 早停
- E5 交叉验证
- E6 bias-variance tradeoff
- E7 特征缩放
- E8 特征选择

## F. 评估指标

- F1 回归：MSE / RMSE / MAE / R2
- F2 分类：Accuracy / Precision / Recall / F1
- F3 ROC-AUC / PR-AUC
- F4 混淆矩阵
- F5 阈值选择
- F6 类别不平衡

## 优先级

```text
必学：A, B, C, D1-D5, E1-E2, F1-F4
次学：D6-D8, E3-E8, F5-F6
```
