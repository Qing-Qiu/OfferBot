# -*- coding: utf-8 -*-
"""
01 NumPy 基础语法可运行示例。

运行方式：
    python 00_Python_Syntax/04_numpy_basics/01_core_syntax_examples.py

学习重点：
    1. ndarray / shape / ndim / dtype
    2. reshape
    3. 索引和切片
    4. 矩阵乘法 @
    5. 广播
    6. 聚合操作
"""

from __future__ import annotations

import numpy as np


def test_array_shape_ndim_dtype() -> None:
    """测试一维与二维数组的形状、维度和类型。"""
    x = np.array([1, 2, 3], dtype=np.float64)
    X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

    assert isinstance(x, np.ndarray)
    assert x.shape == (3,)
    assert x.ndim == 1
    assert x.dtype == np.float64

    assert X.shape == (2, 3)
    assert X.ndim == 2


def test_reshape_changes_shape_not_values() -> None:
    """测试 reshape 只改形状，不改元素内容。"""
    x = np.array([1, 2, 3, 4])
    x2 = x.reshape(2, 2)
    x3 = x.reshape(-1, 1)

    assert x2.shape == (2, 2)
    assert x3.shape == (4, 1)
    assert x2[0, 0] == 1
    assert x2[1, 1] == 4


def test_indexing_and_slicing() -> None:
    """测试常见切片写法。"""
    X = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )

    np.testing.assert_array_equal(X[0], np.array([1, 2, 3]))
    np.testing.assert_array_equal(X[:, 0], np.array([1, 4, 7]))
    np.testing.assert_array_equal(X[:2, :], np.array([[1, 2, 3], [4, 5, 6]]))
    np.testing.assert_array_equal(X[:, 1:3], np.array([[2, 3], [5, 6], [8, 9]]))


def test_matrix_multiplication_matches_linear_regression_intuition() -> None:
    """测试 X @ w 的形状和值，理解线性模型前向计算。"""
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    w = np.array([10.0, 100.0])
    b = 7.0

    y_hat = X @ w + b

    assert y_hat.shape == (3,)
    np.testing.assert_allclose(y_hat, np.array([217.0, 437.0, 657.0]))


def test_broadcasting_adds_row_vector_to_each_row() -> None:
    """测试广播：二维矩阵加一维向量。"""
    X = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    b = np.array([10, 20, 30])

    Y = X + b

    np.testing.assert_array_equal(Y, np.array([[11, 22, 33], [14, 25, 36]]))


def test_aggregation_ops() -> None:
    """测试 sum / mean / max / argmax。"""
    x = np.array([1, 7, 3])

    assert np.sum(x) == 11
    assert np.mean(x) == 11 / 3
    assert np.max(x) == 7
    assert np.argmax(x) == 1


def run_all_tests() -> None:
    tests = [
        test_array_shape_ndim_dtype,
        test_reshape_changes_shape_not_values,
        test_indexing_and_slicing,
        test_matrix_multiplication_matches_linear_regression_intuition,
        test_broadcasting_adds_row_vector_to_each_row,
        test_aggregation_ops,
    ]

    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")


def run_demo() -> None:
    """演示几段最常见的 NumPy 语法。"""
    print("\n=== Demo: NumPy Core Syntax ===")

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    w = np.array([0.5, 1.5])
    b = 2.0

    print(f"X.shape = {X.shape}")
    print(f"w.shape = {w.shape}")
    print(f"X @ w = {X @ w}")
    print(f"X @ w + b = {X @ w + b}")

    x = np.array([1, 2, 3, 4])
    print(f"x.shape = {x.shape}")
    print(f"x.reshape(-1, 1).shape = {x.reshape(-1, 1).shape}")

    Y = np.array([[1, 2, 3], [4, 5, 6]])
    row_bias = np.array([10, 20, 30])
    print(f"broadcast result =\n{Y + row_bias}")


if __name__ == "__main__":
    run_all_tests()
    run_demo()
