# 03 PyTorch 与深度学习知识树

目标：从 Tensor 到训练循环，能手写核心模块并解释反向传播和训练稳定性。

## 建议小章节

```text
01_deep_learning_basics
02_tensor_basics
03_autograd
04_nn_module
05_dataset_dataloader
06_training_loop_and_optim
07_model_components
08_training_engineering
09_interview_playbook
```

说明：上面的小章节适合后续逐目录展开；下面的 A/B/C... 仍然保留为更细粒度知识点树。

## A. 深度学习基础

- A1 神经元：`Wx + b`
- A2 激活函数
- A3 前向传播
- A4 损失函数
- A5 反向传播
- A6 计算图
- A7 梯度下降
- A8 泛化、过拟合、欠拟合

## B. PyTorch Tensor

- B1 Tensor 创建
- B2 shape / dtype / device
- B3 索引、切片
- B4 reshape / view / permute
- B5 广播
- B6 contiguous
- B7 CPU / GPU 迁移
- B8 Tensor 与 ndarray 转换

## C. autograd

- C1 `requires_grad`
- C2 叶子节点
- C3 `grad_fn`
- C4 `backward`
- C5 `.grad`
- C6 梯度累积与 `zero_grad`
- C7 `detach`
- C8 `no_grad`
- C9 in-place 操作风险
- C10 手写 mini autograd

## D. nn.Module

- D1 `nn.Module`
- D2 `__init__` 与 `forward`
- D3 参数注册
- D4 `nn.Linear`
- D5 `nn.Embedding`
- D6 激活函数
- D7 Dropout
- D8 BatchNorm
- D9 LayerNorm / RMSNorm
- D10 自定义层

## E. 训练工程

- E1 Dataset / DataLoader
- E2 train loop
- E3 eval loop
- E4 loss function
- E5 optimizer：SGD / Adam / AdamW
- E6 scheduler
- E7 checkpoint
- E8 seed 与复现
- E9 mixed precision
- E10 梯度裁剪
- E11 日志与指标

## F. 模型专题

- F1 线性层与 MLP
- F2 CNN
- F3 RNN / LSTM / GRU
- F4 Attention
- F5 Transformer Encoder Block
- F6 AutoEncoder
- F7 对比学习基础

## G. 面试重点

- G1 CrossEntropyLoss 为什么输入 logits
- G2 BatchNorm vs LayerNorm
- G3 Adam vs AdamW
- G4 梯度消失/爆炸
- G5 train/eval 差异
- G6 显存估算
- G7 训练不收敛排查

## 优先级

```text
必学：A, B1-B8, C1-C8, D1-D6, E1-E8, G1-G5
次学：C9-C10, D7-D10, E9-E11, F, G6-G7
```
