# -*- coding: utf-8 -*-
"""
01 PyTorch 语法预备：Tensor / autograd / nn.Module 最小示例。

运行方式：
    python 00_Python_Syntax/06_pytorch_syntax_preview/01_tensor_autograd_module_preview.py

说明：
    - 如果当前环境没有安装 torch，本脚本会友好提示并跳过运行。
    - 这样做是为了不阻塞仓库结构整理；后续安装 torch 后可直接复用本脚本。
"""

from __future__ import annotations

import importlib


def _load_torch():
    """延迟导入 torch，避免环境没装时报 ImportError 直接中断整个文件。"""
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return None

    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.optim as optim  # type: ignore

    return torch, nn, optim


def test_tensor_shape_dtype_and_device(torch_module) -> None:
    """Tensor 应该有和 NumPy 类似的 shape，并且携带 dtype/device 信息。"""
    torch = torch_module

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    assert tuple(x.shape) == (2, 2)
    assert x.dtype.is_floating_point
    assert str(x.device) == "cpu"


def test_requires_grad_and_backward(torch_module) -> None:
    """requires_grad=True 的张量应能在 backward 后得到 grad。"""
    torch = torch_module

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x**2).sum()
    y.backward()

    assert y.ndim == 0
    assert x.grad is not None
    expected = torch.tensor([2.0, 4.0, 6.0])
    assert torch.allclose(x.grad, expected)


def test_grad_accumulates_without_zeroing(torch_module) -> None:
    """多次 backward 会累计梯度。"""
    torch = torch_module

    x = torch.tensor([1.0, 2.0], requires_grad=True)

    y1 = (x**2).sum()
    y1.backward()
    first_grad = x.grad.clone()

    y2 = (x**2).sum()
    y2.backward()

    assert torch.allclose(first_grad, torch.tensor([2.0, 4.0]))
    assert torch.allclose(x.grad, torch.tensor([4.0, 8.0]))


def test_nn_module_registers_parameters(torch_module, nn_module) -> None:
    """nn.Module 中的子层参数应能被 parameters() 收集。"""
    torch = torch_module
    nn = nn_module

    class TinyLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x)

    model = TinyLinear()
    params = list(model.parameters())

    assert len(params) == 2  # weight + bias
    x = torch.tensor([[1.0, 2.0]])
    y = model(x)
    assert tuple(y.shape) == (1, 1)


def test_optimizer_step_changes_parameters(torch_module, nn_module, optim_module) -> None:
    """最小训练三连应该能更新参数。"""
    torch = torch_module
    nn = nn_module
    optim = optim_module

    model = nn.Linear(2, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    x = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[3.0]])

    old_weight = model.weight.detach().clone()
    old_bias = model.bias.detach().clone()

    pred = model(x)
    loss = ((pred - target) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    new_weight = model.weight.detach().clone()
    new_bias = model.bias.detach().clone()

    assert not torch.allclose(old_weight, new_weight)
    assert not torch.allclose(old_bias, new_bias)


def run_all_tests() -> bool:
    """运行测试；如果环境没有 torch，则跳过并返回 False。"""
    loaded = _load_torch()
    if loaded is None:
        print("SKIP: 当前环境未安装 torch，PyTorch 示例未执行。")
        return False

    torch, nn, optim = loaded

    tests = [
        lambda: test_tensor_shape_dtype_and_device(torch),
        lambda: test_requires_grad_and_backward(torch),
        lambda: test_grad_accumulates_without_zeroing(torch),
        lambda: test_nn_module_registers_parameters(torch, nn),
        lambda: test_optimizer_step_changes_parameters(torch, nn, optim),
    ]

    for index, test in enumerate(tests, start=1):
        test()
        print(f"[PASS] test_{index}")

    return True


def run_demo() -> None:
    """给出最小可读的 PyTorch 前向与反向例子。"""
    loaded = _load_torch()
    if loaded is None:
        print("SKIP: 当前环境未安装 torch，Demo 未执行。")
        return

    torch, nn, optim = loaded

    print("\n=== Demo: Tensor / autograd / nn.Module ===")

    x = torch.tensor([[1.0, 2.0]], requires_grad=False)
    model = nn.Linear(2, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    target = torch.tensor([[3.0]])

    pred = model(x)
    loss = ((pred - target) ** 2).mean()

    print(f"x.shape = {tuple(x.shape)}")
    print(f"pred.shape = {tuple(pred.shape)}")
    print(f"loss.ndim = {loss.ndim}")

    optimizer.zero_grad()
    loss.backward()

    for name, param in model.named_parameters():
        print(f"{name}.grad.shape = {tuple(param.grad.shape)}")

    optimizer.step()
    print("optimizer.step() 完成")


if __name__ == "__main__":
    run_all_tests()
    run_demo()
