# -*- coding: utf-8 -*-
"""
02 shape 与矩阵乘法专项练习。

运行方式：
    python 00_Python_Syntax/04_numpy_basics/02_shape_and_matmul_drills.py

目标：
    1. 强化 shape 直觉
    2. 区分 (n,) / (n, 1) / (1, n)
    3. 看懂 X @ w + b
    4. 识别广播是否合法
"""

from __future__ import annotations

import numpy as np


def test_matmul_with_vector_returns_1d_output() -> None:
    """(n_samples, n_features) @ (n_features,) -> (n_samples,)"""
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    w = np.array([10.0, 20.0, 30.0])

    y = X @ w

    assert y.shape == (2,)
    np.testing.assert_allclose(y, np.array([140.0, 320.0]))


def test_matmul_with_column_vector_returns_2d_output() -> None:
    """(n_samples, n_features) @ (n_features, 1) -> (n_samples, 1)"""
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    w = np.array([[10.0], [20.0], [30.0]])

    y = X @ w

    assert y.shape == (2, 1)
    np.testing.assert_allclose(y, np.array([[140.0], [320.0]]))


def test_scalar_bias_broadcasts_to_each_prediction() -> None:
    """标量偏置会广播到每个样本。"""
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    w = np.array([1.0, 10.0])
    b = 5.0

    y_hat = X @ w + b

    assert y_hat.shape == (3,)
    np.testing.assert_allclose(y_hat, np.array([26.0, 48.0, 70.0]))


def test_row_vector_broadcasting_adds_to_each_row() -> None:
    """(2, 3) + (3,) 是合法广播。"""
    X = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    row_bias = np.array([10, 20, 30])

    Y = X + row_bias

    assert Y.shape == (2, 3)
    np.testing.assert_array_equal(Y, np.array([[11, 22, 33], [14, 25, 36]]))


def test_mse_loss_reduces_to_scalar() -> None:
    """误差平方后取 mean，最终应该得到标量。"""
    y_pred = np.array([2.0, 4.0, 6.0])
    y_true = np.array([1.0, 5.0, 7.0])

    errors = y_pred - y_true
    loss = np.mean(errors**2)

    assert errors.shape == (3,)
    assert np.isscalar(loss)
    assert loss == 1.0


def test_invalid_broadcasting_example_raises_error() -> None:
    """(2, 4) + (2,) 通常不合法，应抛错。"""
    X = np.zeros((2, 4))
    b = np.array([1.0, 2.0])

    try:
        _ = X + b
    except ValueError:
        return

    raise AssertionError("非法广播没有抛出 ValueError。")


def run_all_tests() -> None:
    tests = [
        test_matmul_with_vector_returns_1d_output,
        test_matmul_with_column_vector_returns_2d_output,
        test_scalar_bias_broadcasts_to_each_prediction,
        test_row_vector_broadcasting_adds_to_each_row,
        test_mse_loss_reduces_to_scalar,
        test_invalid_broadcasting_example_raises_error,
    ]

    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")


def run_demo() -> None:
    print("\n=== Demo: Shape And Matmul Drills ===")

    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    w_1d = np.array([10.0, 20.0, 30.0])
    w_2d = np.array([[10.0], [20.0], [30.0]])

    y1 = X @ w_1d
    y2 = X @ w_2d

    print(f"X.shape = {X.shape}")
    print(f"w_1d.shape = {w_1d.shape}, (X @ w_1d).shape = {y1.shape}")
    print(f"w_2d.shape = {w_2d.shape}, (X @ w_2d).shape = {y2.shape}")

    bias = 5.0
    print(f"X @ w_1d + bias = {y1 + bias}")


if __name__ == "__main__":
    run_all_tests()
    run_demo()
