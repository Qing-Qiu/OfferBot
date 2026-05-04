# 04 Information Theory：信息论基础

## 核心直觉

信息论关心的是“不确定性”和“编码代价”。越罕见的事件，发生时带来的信息量越大；一个分布越平均，不确定性越高。

在 AI 里，交叉熵、KL 散度、困惑度都在回答同一个问题：

```text
模型预测的概率分布，和真实分布差多远
```

## 必要公式

信息量：

```text
I(x) = -log P(x)
```

熵：

```text
H(P) = -sum_x P(x) log P(x)
```

交叉熵：

```text
H(P, Q) = -sum_x P(x) log Q(x)
```

KL 散度：

```text
KL(P || Q) = sum_x P(x) log(P(x) / Q(x))
```

关系式：

```text
H(P, Q) = H(P) + KL(P || Q)
```

分类交叉熵 one-hot 形式：

```text
L = -log p_true_class
```

困惑度：

```text
PPL = exp(average negative log likelihood)
```

## AI 用法

分类任务中，真实标签通常是 one-hot 分布。此时交叉熵会退化成：

```text
把真实类别的预测概率取负 log
```

语言模型训练时，目标是预测下一个 token。每个位置都有一个 vocabulary 上的概率分布，loss 通常就是 token-level cross entropy。

KL 散度常用于蒸馏、VAE、RLHF、DPO 等场景，用来约束一个分布不要偏离另一个分布太远。

困惑度常用于语言模型评估。PPL 越低，说明模型平均给真实 token 的概率越高。

## 面试陷阱

- KL 散度不是距离，因为 `KL(P || Q) != KL(Q || P)`。
- 交叉熵不是只用于分类；语言模型也是在做大量 token 分类。
- 熵衡量真实分布自己的不确定性，交叉熵衡量用预测分布编码真实分布的代价。
- PPL 低通常更好，但它不等价于所有下游任务都更好。
- softmax 输出概率，cross entropy 负责惩罚真实类别概率太低。

## 自测题

1. 为什么小概率事件的信息量更大？
2. 熵高说明分布更确定还是更不确定？
3. one-hot 标签下，交叉熵为什么等于 `-log p_true_class`？
4. KL 散度为什么不是真正的距离？
5. `H(P, Q) = H(P) + KL(P || Q)` 说明优化交叉熵时在优化什么？
6. 语言模型的 cross entropy 是在预测什么？
7. 困惑度和 average negative log likelihood 有什么关系？
8. RLHF / DPO 里为什么常出现 KL 约束或 KL 直觉？
