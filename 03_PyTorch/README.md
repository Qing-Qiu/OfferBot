# 03 PyTorch：深度学习实现地基

本章负责把机器学习的训练闭环落到 PyTorch 代码里。学习目标不是只会调库，而是能读懂：

```text
Tensor -> autograd -> nn.Module -> DataLoader -> training loop -> optimizer
```

## 当前内容

- [01_deep_learning_foundations.md](./01_deep_learning_foundations.md)：深度学习训练闭环总览。
- [02_tensor_basics](./02_tensor_basics/README.md)：Tensor、shape、dtype、device、view/reshape/permute、broadcast、contiguous。
- [03_autograd](./03_autograd/README.md)：计算图、叶子节点、`grad_fn`、`backward`、梯度累积、`detach`、`no_grad`。
- [06_training_loop_and_optim](./06_training_loop_and_optim/README.md)：训练循环、eval 循环、optimizer、scheduler、checkpoint、seed。

## 推荐学习顺序

1. 先读 `01_deep_learning_foundations.md`，建立训练闭环整体感。
2. 再读 Tensor，掌握 shape、dtype、device。
3. 再读 autograd，理解 `loss.backward()` 到底在做什么。
4. 最后读训练循环，把 `zero_grad -> backward -> step` 连成肌肉记忆。
