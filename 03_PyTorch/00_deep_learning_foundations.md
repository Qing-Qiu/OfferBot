# 00 深度学习基础：从最小训练闭环开始

> 学习目标：理解深度学习到底在“学”什么，训练循环每一步为什么存在，以及面试中如何把模型、损失、梯度、反向传播、优化和泛化讲成一条完整逻辑链。

## 0. 一句话理解深度学习

深度学习的核心不是“堆很多层”，而是：

```text
用带参数的函数 f(x; theta) 去拟合输入 x 到目标 y 的映射，
通过损失函数衡量预测错误，
再用梯度下降不断调整参数 theta，
让模型在未见过的数据上也表现好。
```

也就是这条最小闭环：

```text
数据 -> 前向传播 -> 预测值 -> 损失函数 -> 反向传播 -> 梯度 -> 优化器更新参数 -> 再次预测
```

只要你真正理解这条链，后面的 CNN、RNN、Transformer、推荐系统排序模型、大模型微调，本质上都只是这条链的不同结构和工程放大版。

## 1. 先分清几个基本概念

| 概念 | 含义 | 面试易错点 |
| --- | --- | --- |
| 样本 sample | 一条训练数据，比如一张图片、一条用户行为、一段文本 | 样本不是特征，样本通常包含输入和标签 |
| 特征 feature | 模型输入中的可用信息，比如像素、词 ID、用户年龄 | 特征质量经常比模型结构更重要 |
| 标签 label | 监督学习中的标准答案 | 真实业务标签可能有噪声和偏差 |
| 参数 parameter | 模型训练出来的量，比如权重 W、偏置 b | 参数由训练学习得到 |
| 超参数 hyperparameter | 人手动设定的量，比如学习率、batch size、层数 | 超参数不是训练直接学出来的 |
| logits | 分类模型 softmax 之前的原始分数 | PyTorch 的 CrossEntropyLoss 通常接收 logits，不要先手动 softmax |
| loss | 当前预测和标签之间的可优化误差 | loss 下降不等于业务指标一定上涨 |
| metric | 评估指标，比如 accuracy、AUC、NDCG | 很多 metric 不可导，所以不能直接作为训练 loss |
| gradient | loss 对参数的导数，表示参数改变时 loss 怎么变 | 梯度方向不是参数更新方向，负梯度方向才是下降方向 |
| epoch | 完整遍历一遍训练集 | epoch 多不一定好，可能过拟合 |
| batch | 一次参数更新使用的一小批样本 | batch size 会影响稳定性、速度和泛化 |

## 2. 监督学习的数学形式

假设训练集为：

```text
D = {(x_1, y_1), (x_2, y_2), ..., (x_m, y_m)}
```

模型是一个带参数的函数：

```text
y_hat = f(x; theta)
```

其中：

- `x` 是输入。
- `y_hat` 是模型预测。
- `y` 是真实标签。
- `theta` 是模型参数，比如所有层的权重和偏置。

训练目标通常写成：

```text
J(theta) = (1 / m) * sum_i L(f(x_i; theta), y_i)
```

如果加上正则化，可以写成：

```text
J(theta) = (1 / m) * sum_i L(f(x_i; theta), y_i) + lambda * R(theta)
```

这就是深度学习的第一条主线：我们不直接“让模型变聪明”，而是定义一个可优化目标 `J(theta)`，再让优化器不断降低它。

## 3. 一个神经元到底在做什么

最简单的神经元可以写成：

```text
z = w^T x + b
a = sigma(z)
```

其中：

- `w^T x + b` 是线性变换。
- `sigma` 是非线性激活函数。
- `a` 是这一层输出。

如果没有非线性激活函数，多层线性层叠在一起仍然等价于一层线性变换：

```text
W3 * (W2 * (W1 * x)) = W_total * x
```

所以“深度”本身不是魔法，非线性才让网络具备表达复杂函数的能力。

## 4. 前向传播：从输入到预测

前向传播就是把输入 `x` 按照模型结构一层层算到输出：

```text
x -> Linear -> Activation -> Linear -> ... -> logits / prediction
```

例如一个二分类 MLP：

```text
h = ReLU(W1 x + b1)
logit = W2 h + b2
p = sigmoid(logit)
```

对于多分类任务，模型一般输出每个类别的 logits：

```text
logits = [score_class_0, score_class_1, ..., score_class_k]
```

注意：logits 不是概率，它们只是未归一化的分数。经过 softmax 后才变成概率分布。

## 5. 损失函数：模型到底错了多少

不同任务对应不同 loss。

### 回归任务：MSE

```text
L = (y_hat - y)^2
```

适合预测连续值，比如房价、温度、点击时长。

### 二分类任务：BCE

```text
L = -[y * log(p) + (1 - y) * log(1 - p)]
```

适合预测 0/1，比如是否点击、是否转化。

### 多分类任务：Cross Entropy

如果真实类别是 `c`，模型给真实类别的概率是 `p_c`：

```text
L = -log(p_c)
```

直觉：模型越不相信正确类别，loss 越大。

面试陷阱：分类任务里的 accuracy 不能直接作为常规梯度训练的 loss，因为 argmax 这类操作不可导。训练通常优化交叉熵，再用 accuracy、AUC 等指标评估。

## 6. 反向传播：链式法则的工程化

反向传播回答的问题是：

```text
每个参数对最终 loss 负了多少责任？
```

假设：

```text
z = w * x + b
y_hat = z
L = (y_hat - y)^2
```

那么：

```text
dL/dw = dL/dy_hat * dy_hat/dz * dz/dw
```

这就是链式法则。神经网络只是把很多这样的局部导数组合成一张计算图。

PyTorch 的 `autograd` 会在前向计算时记录张量操作形成计算图，在调用 `loss.backward()` 时自动沿图反向计算梯度。

## 7. 梯度下降：参数如何更新

最基础的参数更新公式：

```text
theta = theta - learning_rate * gradient
```

其中：

- `gradient` 指向 loss 上升最快的方向。
- `-gradient` 指向 loss 下降最快的局部方向。
- `learning_rate` 控制每一步走多远。

学习率太小：训练很慢，可能长时间看不到效果。

学习率太大：loss 震荡甚至发散。

## 8. 训练循环的标准步骤

一个最小 PyTorch 训练循环通常长这样：

```text
for epoch in range(num_epochs):
    for x, y in dataloader:
        pred = model(x)             # 1. 前向传播
        loss = loss_fn(pred, y)     # 2. 计算损失
        optimizer.zero_grad()       # 3. 清空旧梯度
        loss.backward()             # 4. 反向传播，计算新梯度
        optimizer.step()            # 5. 根据梯度更新参数
```

为什么要 `zero_grad()`？

PyTorch 中梯度默认会累积到参数的 `.grad` 属性上。如果每个 batch 前不清零，当前 batch 的梯度会和之前 batch 的梯度混在一起，导致更新错误，除非你是在刻意做梯度累积。

## 9. 泛化：为什么训练集好不代表真好

训练时我们能直接优化的是训练集上的经验损失：

```text
training loss = 训练样本上的平均 loss
```

但真正关心的是模型在未知数据上的表现：

```text
generalization = 未见过样本上的表现
```

所以数据通常拆成：

- 训练集：用于更新参数。
- 验证集：用于调超参数、早停、选择模型。
- 测试集：用于最终评估，尽量不要反复看。

### 欠拟合

表现：

- 训练集 loss 高。
- 验证集 loss 也高。

常见原因：

- 模型太简单。
- 特征不足。
- 训练不够。
- 学习率设置不合理。

### 过拟合

表现：

- 训练集 loss 很低。
- 验证集 loss 高或开始变差。

常见原因：

- 模型容量过大。
- 数据太少。
- 标签噪声。
- 正则化不足。
- 数据泄漏或切分不合理。

## 10. 面试常考陷阱

### 陷阱 1：loss 和 metric 混为一谈

loss 是训练优化目标，metric 是评估业务效果或模型效果的指标。它们可以相关，但不是同一个东西。

### 陷阱 2：多分类先 softmax 再传 CrossEntropyLoss

PyTorch 的 `nn.CrossEntropyLoss` 会在内部组合 log-softmax 和负对数似然，通常应该传 logits。如果先手动 softmax，可能带来数值稳定性和梯度问题。

### 陷阱 3：忘记清空梯度

`loss.backward()` 会把梯度累积到 `.grad`。普通训练循环里要在每次更新前后合理调用 `optimizer.zero_grad()`。

### 陷阱 4：训练和评估模式不切换

含 Dropout、BatchNorm 的模型中：

- `model.train()` 用于训练。
- `model.eval()` 用于验证和测试。

忘记切换会导致评估结果不稳定或不可信。

### 陷阱 5：只看训练集表现

训练集表现好只能说明模型记住或拟合了训练数据，不代表泛化好。面试时一定要主动提验证集、测试集、过拟合和早停。

### 陷阱 6：数据预处理泄漏

标准化、归一化、分桶等统计量应该只在训练集上拟合，再应用到验证集和测试集。否则验证和测试信息会泄漏到训练过程。

## 11. 面试高频问法

### Q1：深度学习训练的完整流程是什么？

答题主线：

```text
定义模型 -> 前向传播 -> 计算损失 -> 反向传播求梯度 -> 优化器更新参数 -> 验证集评估 -> 重复直到收敛或早停
```

加分点：主动提到训练集、验证集、测试集的职责不同。

### Q2：为什么神经网络需要激活函数？

没有激活函数，多层线性网络仍然等价于一层线性变换，表达能力有限。激活函数引入非线性，使模型能够拟合复杂函数。

### Q3：反向传播的本质是什么？

反向传播的本质是链式法则在计算图上的高效应用。每个节点只需要保存局部导数，最终从 loss 反向传播到所有参数。

### Q4：学习率过大或过小会怎样？

学习率过小会导致收敛慢，甚至停在不理想区域；学习率过大可能导致 loss 震荡、发散，越过较优解。

### Q5：为什么训练 loss 下降但测试效果不好？

可能是过拟合、训练测试分布不一致、数据泄漏、标签噪声、metric 与 loss 不一致，或者验证/测试集切分方式不合理。

## 12. 第一阶段你要掌握到什么程度

你不需要一开始就会推所有复杂公式，但至少要能闭眼讲清楚：

- 模型参数是什么。
- loss 为什么能代表“错了多少”。
- 梯度为什么能指导参数更新。
- 反向传播为什么是链式法则。
- 训练集、验证集、测试集各自干什么。
- 过拟合和欠拟合如何判断。
- 一个最小训练循环有哪些步骤。

## 13. 接下来三节课建议

### 第 1 节：手写线性回归

目标：用 NumPy 从零实现 `y = wx + b` 的前向、MSE、梯度和参数更新。严格分类上，线性回归属于经典机器学习基础，因此代码放在 `02_ML_Foundations`；它在这里作为深度学习训练闭环的最小例子。

### 第 2 节：手写二分类逻辑回归

目标：理解 sigmoid、BCE、分类阈值、logits 和概率的区别。

### 第 3 节：PyTorch autograd 入门

目标：把前两节手写梯度的过程交给 PyTorch，理解 `requires_grad`、`backward()`、`.grad`、`zero_grad()`。

## 14. 参考资料

- [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)：PyTorch 官方基础教程，覆盖数据、模型、优化和保存加载流程。
- [PyTorch Autograd Tutorial](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)：PyTorch 官方自动微分教程。
- [PyTorch Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)：PyTorch 官方优化循环教程。
- [Deep Learning Book, Chapter 8: Optimization for Training Deep Models](https://www.deeplearningbook.org/contents/optimization.html)：深度学习优化与经验风险、泛化目标的经典教材章节。
- [Google Machine Learning Crash Course: Neural Networks](https://developers.google.com/machine-learning/crash-course/neural-networks)：Google 官方机器学习速成课程中的神经网络入门。
