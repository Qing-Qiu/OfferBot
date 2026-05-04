# 06 Training Loop And Optim：训练循环与优化器

## 核心直觉

训练循环是深度学习工程的主干。模型再复杂，最终都会落到这几步：

```text
取数据
前向计算
计算 loss
清梯度
反向传播
更新参数
记录指标
保存 checkpoint
```

面试里如果让你手写 PyTorch 训练代码，通常不是考 API 背诵，而是考你是否知道每一步为什么存在。

## 关键概念

`Dataset`：定义单条样本怎么取。

`DataLoader`：负责 batch、shuffle、多进程加载。

`model.train()`：打开训练模式，影响 Dropout、BatchNorm 等层。

`model.eval()`：打开评估模式，关闭 Dropout 随机性，BatchNorm 使用移动统计量。

`optimizer`：根据梯度更新参数，例如 SGD、Adam、AdamW。

`scheduler`：调整学习率。

`checkpoint`：保存模型参数、优化器状态、epoch、随机种子等，支持恢复训练。

`seed`：控制随机性，提升实验可复现性。

## 必要公式 / shape / 流程

标准训练循环：

```python
for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
```

标准评估循环：

```python
model.eval()
with torch.no_grad():
    for x, y in valid_loader:
        logits = model(x)
        metric.update(logits, y)
```

AdamW 更新直觉：

```text
梯度方向来自 loss
一阶矩平滑方向
二阶矩调节步长
weight decay 控制参数规模
```

checkpoint 内容：

```text
model.state_dict()
optimizer.state_dict()
scheduler.state_dict()
epoch
best_metric
config
```

## 代码阅读提示

读训练代码时按顺序检查：

```text
数据是否进 device
train/eval 模式是否正确
zero_grad 是否在 backward 前
loss 输入和 target shape 是否匹配
optimizer.step 是否在 backward 后
eval 是否包了 no_grad
指标是否和 loss 分开统计
checkpoint 是否保存了优化器状态
```

如果训练不收敛，优先排查：学习率、数据标签、loss 使用方式、梯度是否为 0 或 NaN、模型是否真的注册参数、train/eval 是否误用。

## 面试高频问法

1. PyTorch 标准训练循环有哪些步骤？
2. `model.train()` 和 `model.eval()` 有什么区别？
3. 为什么评估时要用 `torch.no_grad()`？
4. `optimizer.zero_grad()` 应该放在哪里？
5. SGD、Adam、AdamW 的区别是什么？
6. checkpoint 应该保存哪些内容？
7. 如何保证训练可复现？
8. 训练不收敛你会怎么排查？

## 常见陷阱

- 只保存模型参数，不保存优化器状态，恢复训练后学习率和动量状态丢失。
- eval 阶段忘记 `model.eval()`，Dropout 和 BatchNorm 行为错误。
- eval 阶段忘记 `no_grad()`，显存占用变大。
- 忘记把输入和标签移动到同一个 device。
- 指标统计没有按样本数加权，最后平均不准。
- scheduler 调用时机不清楚，不同 scheduler 可能按 step 或按 epoch 更新。

## 自测题

1. 写出 PyTorch 训练循环的五个核心调用。
2. 为什么 `zero_grad()` 要在 `backward()` 前调用？
3. `model.eval()` 会停止 autograd 吗？
4. `torch.no_grad()` 会改变 Dropout 行为吗？
5. checkpoint 只保存 `model.state_dict()` 够不够？为什么？
6. AdamW 相比 Adam + L2 有什么直觉区别？
7. 训练 loss 变成 NaN，优先排查什么？
8. 如何确认模型参数真的被 optimizer 管理？
