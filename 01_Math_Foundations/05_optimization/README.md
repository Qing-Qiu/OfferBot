# 05 Optimization：优化理论基础

## 核心直觉

优化关心的是：给定一个目标函数，怎么找到让它尽量小的参数。机器学习里的目标函数通常就是 loss。

深度学习优化和传统凸优化不同：神经网络大多是非凸的，我们通常不追求证明全局最优，而是用稳定的训练策略找到泛化较好的参数。

## 必要公式

目标函数：

```text
min_theta L(theta)
```

凸函数直觉：

```text
任意两点连线都在函数图像上方
局部最优就是全局最优
```

非凸函数直觉：

```text
可能有多个局部最优、平坦区和鞍点
深度网络通常是非凸优化
```

梯度下降：

```text
theta_t+1 = theta_t - lr * grad L(theta_t)
```

SGD：

```text
theta_t+1 = theta_t - lr * grad L_batch(theta_t)
```

Momentum：

```text
v_t = beta v_t-1 + grad L(theta_t)
theta_t+1 = theta_t - lr * v_t
```

Adam 直觉：

```text
m_t: 梯度一阶矩，类似动量
v_t: 梯度平方二阶矩，调节每个参数的步长
```

L2 正则：

```text
L_total = L_data + lambda ||theta||_2^2
```

L1 正则：

```text
L_total = L_data + lambda ||theta||_1
```

约束优化与拉格朗日乘子：

```text
min_x f(x), subject to g(x) = 0
Lagrangian = f(x) + lambda g(x)
```

直觉：把“必须满足的约束”合进一个新的目标函数里一起优化。

log-sum-exp：

```text
log sum_i exp(x_i)
= m + log sum_i exp(x_i - m)
m = max_i x_i
```

## AI 用法

SGD 和 mini-batch SGD 是深度学习训练的基础。mini-batch 梯度有噪声，但计算更快，也可能帮助模型跳出一些不好的区域。

Momentum 用历史梯度平滑更新方向。Adam / AdamW 进一步给不同参数自适应学习率，实践中非常常用。

AdamW 把 weight decay 和 Adam 的梯度更新解耦，通常比把 L2 正则直接塞进 Adam 更稳。

数值稳定性在 softmax、cross entropy、log probability 中极其重要。实际框架通常会把 softmax 和 log 合并成稳定实现。

## 面试陷阱

- 学习率太大可能震荡或发散，太小会收敛慢。
- SGD 的梯度是 mini-batch 上的估计，不是全量真实梯度。
- Adam 收敛快不代表一定泛化最好。
- L1 更容易产生稀疏参数，L2 更倾向于让参数整体变小。
- AdamW 的 weight decay 不是简单等价于 Adam + L2。
- softmax 直接 `exp(logits)` 可能 overflow。

## 自测题

1. 梯度下降为什么沿负梯度方向更新？
2. mini-batch SGD 和 full-batch GD 的区别是什么？
3. Momentum 解决了什么问题？
4. Adam 的一阶矩和二阶矩分别是什么直觉？
5. AdamW 和 Adam + L2 的关键区别是什么？
6. L1 和 L2 正则对参数有什么不同影响？
7. 为什么 softmax 需要数值稳定技巧？
8. 什么是鞍点？它和局部最优有什么区别？
9. 学习率过大和过小分别会怎样？
10. 为什么凸优化里局部最优就是全局最优？
11. 拉格朗日乘子想解决什么类型的问题？
