# -*- coding: utf-8 -*-
"""
01 NumPy 手写线性回归：从零理解前向传播、MSE、梯度下降和训练循环。

运行方式：
    python 02_ML_Foundations/01_numpy_linear_regression_from_scratch.py

本文件刻意不使用 sklearn / PyTorch，目的不是追求最少代码，
而是把深度学习训练闭环拆开给你看：

    数据 -> 前向传播 -> 损失函数 -> 反向传播/求梯度 -> 参数更新 -> 评估

线性回归虽然简单，但它已经包含了深度学习训练中最核心的机制。
后面学神经网络时，只是把这里的线性函数 f(x)=Wx+b 换成更复杂的函数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TrainingHistory:
    """保存训练过程中的关键信息，方便排查 loss 是否正常下降。"""

    losses: list[float]
    epochs_run: int


class NumpyLinearRegression:
    """
    使用批量梯度下降训练线性回归模型。

    模型形式：
        y_hat = X @ w + b

    其中：
        X: 输入特征矩阵，形状为 (n_samples, n_features)
        w: 权重向量，形状为 (n_features,)
        b: 偏置标量
        y_hat: 预测值，形状为 (n_samples,)

    训练目标：
        最小化均方误差 MSE = mean((y_hat - y) ** 2)

    注意：
        这里使用的是 full-batch gradient descent，也就是每次参数更新都看完整训练集。
        真实深度学习训练通常使用 mini-batch gradient descent。
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_epochs: int = 1000,
        fit_intercept: bool = True,
        tol: float = 0.0,
        verbose: bool = False,
    ) -> None:
        """
        初始化线性回归模型。

        Args:
            learning_rate: 学习率，控制每次沿负梯度方向走多远。
            n_epochs: 训练轮数，每一轮都会完整使用一次训练集更新参数。
            fit_intercept: 是否学习偏置 b。通常应该为 True。
            tol: 早停阈值。如果相邻两轮 loss 改变量小于 tol，则提前停止。
            verbose: 是否打印训练过程中的 loss。
        """
        if learning_rate <= 0:
            raise ValueError("learning_rate 必须大于 0。")
        if not isinstance(n_epochs, int) or n_epochs < 0:
            raise ValueError("n_epochs 必须是非负整数。")
        if tol < 0:
            raise ValueError("tol 必须大于等于 0。")

        self.learning_rate = float(learning_rate)
        self.n_epochs = n_epochs
        self.fit_intercept = fit_intercept
        self.tol = float(tol)
        self.verbose = verbose

        # 约定：带下划线的属性表示 fit 之后才会被学习出来。
        self.weight_: Optional[np.ndarray] = None
        self.bias_: Optional[float] = None
        self.history_: Optional[TrainingHistory] = None

    @staticmethod
    def _validate_x(X: np.ndarray) -> np.ndarray:
        """
        校验并规范化特征矩阵 X。

        线性模型要求 X 是二维矩阵：
            - 一行代表一个样本。
            - 一列代表一个特征。

        如果用户传入一维数组，例如 [1, 2, 3]，这里会自动转成
        [[1], [2], [3]]，也就是 3 个样本、1 个特征。
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X 必须是一维或二维数组。")
        if X.shape[0] == 0:
            raise ValueError("X 至少要包含一个样本。")
        if X.shape[1] == 0:
            raise ValueError("X 至少要包含一个特征。")
        if not np.all(np.isfinite(X)):
            raise ValueError("X 不能包含 NaN 或 Inf。")

        return X

    @staticmethod
    def _validate_y(y: np.ndarray) -> np.ndarray:
        """
        校验并规范化标签向量 y。

        线性回归中的 y 应该是一维向量：
            y.shape == (n_samples,)

        如果用户传入形状为 (n_samples, 1) 的二维列向量，这里会压平成一维。
        """
        y = np.asarray(y, dtype=np.float64)

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)

        if y.ndim != 1:
            raise ValueError("y 必须是一维数组，或形状为 (n_samples, 1) 的列向量。")
        if y.shape[0] == 0:
            raise ValueError("y 至少要包含一个标签。")
        if not np.all(np.isfinite(y)):
            raise ValueError("y 不能包含 NaN 或 Inf。")

        return y

    @classmethod
    def _validate_xy(cls, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """同时校验 X 和 y，并确保样本数一致。"""
        X = cls._validate_x(X)
        y = cls._validate_y(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X 和 y 的样本数必须一致，但得到 X.shape[0]={X.shape[0]}, "
                f"y.shape[0]={y.shape[0]}。"
            )

        return X, y

    def _check_is_fitted(self) -> None:
        """预测或计算梯度前，必须先调用 fit 学到参数。"""
        if self.weight_ is None or self.bias_ is None:
            raise RuntimeError("模型尚未训练，请先调用 fit(X, y)。")

    def _initialize_parameters(self, n_features: int) -> None:
        """
        初始化参数。

        线性回归使用零初始化是可以的；对普通神经网络来说，
        多层权重通常不能全部零初始化，否则不同神经元会学到相同东西。
        """
        self.weight_ = np.zeros(n_features, dtype=np.float64)
        self.bias_ = 0.0

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        前向传播：根据当前参数计算预测值。

        y_hat = X @ w + b
        """
        self._check_is_fitted()
        return X @ self.weight_ + self.bias_

    @staticmethod
    def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        均方误差 MSE。

        MSE = mean((y_pred - y_true) ** 2)

        这个函数越小，说明当前预测值和真实标签整体越接近。
        """
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)

        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred 和 y_true 的形状必须一致。")
        if y_pred.size == 0:
            raise ValueError("y_pred 和 y_true 不能为空。")
        if not np.all(np.isfinite(y_pred)) or not np.all(np.isfinite(y_true)):
            raise ValueError("y_pred 和 y_true 不能包含 NaN 或 Inf。")

        errors = y_pred - y_true
        return float(np.mean(errors**2))

    def _compute_loss_and_gradients(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[float, np.ndarray, float]:
        """
        计算当前 loss 以及 loss 对参数 w、b 的梯度。

        模型：
            y_hat = X @ w + b

        损失：
            L = mean((y_hat - y) ** 2)

        对 w 的梯度：
            dL/dw = (2 / n) * X.T @ (y_hat - y)

        对 b 的梯度：
            dL/db = (2 / n) * sum(y_hat - y)

        这一步就是本例中的“反向传播”。在更复杂的神经网络里，
        PyTorch autograd 会沿着计算图自动完成同类链式求导。
        """
        self._check_is_fitted()

        n_samples = X.shape[0]
        y_pred = self._forward(X)
        errors = y_pred - y
        loss = self.mse_loss(y_pred, y)

        grad_w = (2.0 / n_samples) * (X.T @ errors)
        grad_b = float(2.0 * np.mean(errors)) if self.fit_intercept else 0.0

        return loss, grad_w, grad_b

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyLinearRegression":
        """
        训练模型。

        每一轮训练做四件事：
            1. 前向传播，计算预测值。
            2. 用预测值和真实值计算 MSE loss。
            3. 根据 MSE 对参数求梯度。
            4. 沿负梯度方向更新参数。
        """
        X, y = self._validate_xy(X, y)
        self._initialize_parameters(n_features=X.shape[1])

        losses: list[float] = []
        previous_loss: Optional[float] = None
        epochs_run = 0

        for epoch in range(self.n_epochs):
            loss, grad_w, grad_b = self._compute_loss_and_gradients(X, y)
            losses.append(loss)
            epochs_run = epoch + 1

            # 参数更新公式：
            #     theta = theta - learning_rate * gradient
            # 注意是减去梯度，因为梯度方向是 loss 上升最快的方向。
            self.weight_ = self.weight_ - self.learning_rate * grad_w
            self.bias_ = self.bias_ - self.learning_rate * grad_b

            if self.verbose and (epoch == 0 or (epoch + 1) % 100 == 0):
                print(f"epoch={epoch + 1:04d}, loss={loss:.8f}")

            # 简单早停：如果 loss 改善非常小，就认为已经基本收敛。
            if previous_loss is not None and abs(previous_loss - loss) < self.tol:
                break
            previous_loss = loss

        self.history_ = TrainingHistory(losses=losses, epochs_run=epochs_run)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用训练好的模型进行预测。

        Args:
            X: 特征矩阵，形状为 (n_samples, n_features)。

        Returns:
            预测值向量，形状为 (n_samples,)。
        """
        self._check_is_fitted()
        X = self._validate_x(X)

        if X.shape[1] != self.weight_.shape[0]:
            raise ValueError(
                f"特征数不匹配：模型需要 {self.weight_.shape[0]} 个特征，"
                f"但输入 X 有 {X.shape[1]} 个特征。"
            )

        return self._forward(X)


def make_synthetic_linear_data(
    n_samples: int = 200,
    noise_std: float = 0.2,
    random_seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    构造一个可控的线性回归数据集，用于演示。

    真实关系：
        y = 3.0 * x1 - 2.0 * x2 + 1.0 + noise

    因为我们知道真实的 w 和 b，所以可以观察模型是否学到了接近的参数。
    """
    if n_samples <= 0:
        raise ValueError("n_samples 必须大于 0。")
    if noise_std < 0:
        raise ValueError("noise_std 必须大于等于 0。")

    rng = np.random.default_rng(random_seed)
    true_w = np.array([3.0, -2.0], dtype=np.float64)
    true_b = 1.0

    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, 2))
    noise = rng.normal(loc=0.0, scale=noise_std, size=n_samples)
    y = X @ true_w + true_b + noise

    return X, y, true_w, true_b


def _assert_allclose(
    actual: np.ndarray | float,
    expected: np.ndarray | float,
    atol: float,
    message: str,
) -> None:
    """测试辅助函数：让断言失败信息更像面试中的定位线索。"""
    if not np.allclose(actual, expected, atol=atol):
        raise AssertionError(f"{message}\nactual={actual}\nexpected={expected}\natol={atol}")


def test_single_feature_recovers_parameters() -> None:
    """普通用例：单特征、无噪声，应该能学回接近真实的 w 和 b。"""
    X = np.linspace(-2.0, 2.0, 100).reshape(-1, 1)
    y = 3.0 * X.reshape(-1) + 2.0

    model = NumpyLinearRegression(learning_rate=0.05, n_epochs=3000, tol=1e-14)
    model.fit(X, y)

    _assert_allclose(model.weight_, np.array([3.0]), atol=1e-6, message="单特征权重没有收敛到真实值。")
    _assert_allclose(model.bias_, 2.0, atol=1e-6, message="单特征偏置没有收敛到真实值。")

    final_loss = model.mse_loss(model.predict(X), y)
    if final_loss > 1e-10:
        raise AssertionError(f"无噪声单特征数据的最终 loss 过大：{final_loss}")


def test_multi_feature_recovers_parameters() -> None:
    """普通用例：多特征、无噪声，应该能学回真实参数。"""
    rng = np.random.default_rng(123)
    X = rng.normal(size=(500, 3))
    true_w = np.array([1.5, -2.0, 0.7])
    true_b = -0.3
    y = X @ true_w + true_b

    model = NumpyLinearRegression(learning_rate=0.05, n_epochs=2500, tol=1e-14)
    model.fit(X, y)

    _assert_allclose(model.weight_, true_w, atol=1e-6, message="多特征权重没有收敛到真实值。")
    _assert_allclose(model.bias_, true_b, atol=1e-6, message="多特征偏置没有收敛到真实值。")


def test_constant_target_learns_bias() -> None:
    """
    边界用例：所有标签都是同一个常数。

    合理结果：
        - 如果 fit_intercept=True，模型应该主要通过 bias 学到这个常数。
        - 权重应该接近 0。
    """
    rng = np.random.default_rng(11)
    X = rng.normal(size=(120, 4))
    y = np.full(shape=120, fill_value=5.0)

    model = NumpyLinearRegression(learning_rate=0.05, n_epochs=2500, tol=1e-14)
    model.fit(X, y)

    _assert_allclose(model.weight_, np.zeros(4), atol=1e-5, message="常数标签场景下权重应该接近 0。")
    _assert_allclose(model.bias_, 5.0, atol=1e-5, message="常数标签场景下偏置应该接近标签常数。")


def test_zero_epoch_only_initializes_parameters() -> None:
    """
    边界用例：n_epochs=0。

    这表示只初始化参数但不训练。模型可以 predict，但参数仍然是初始值。
    """
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([10.0, 20.0, 30.0])

    model = NumpyLinearRegression(learning_rate=0.1, n_epochs=0)
    model.fit(X, y)

    _assert_allclose(model.weight_, np.array([0.0]), atol=0.0, message="零轮训练时权重应保持初始值。")
    _assert_allclose(model.bias_, 0.0, atol=0.0, message="零轮训练时偏置应保持初始值。")
    _assert_allclose(model.predict(X), np.zeros(3), atol=0.0, message="零轮训练时预测应来自初始参数。")


def test_predict_before_fit_raises_error() -> None:
    """边界用例：未训练就预测，应该给出明确错误。"""
    model = NumpyLinearRegression()

    try:
        model.predict(np.array([[1.0]]))
    except RuntimeError:
        return

    raise AssertionError("未调用 fit 就 predict 应该抛出 RuntimeError。")


def test_invalid_inputs_raise_errors() -> None:
    """边界用例：非法输入必须尽早报错，避免静默产生错误训练结果。"""
    invalid_cases = [
        (np.empty((0, 2)), np.array([]), "空样本"),
        (np.array([[1.0], [2.0]]), np.array([1.0]), "X/y 样本数不一致"),
        (np.array([[np.nan]]), np.array([1.0]), "X 包含 NaN"),
        (np.array([[1.0]]), np.array([np.inf]), "y 包含 Inf"),
        (np.ones((2, 2, 2)), np.array([1.0, 2.0]), "X 维度非法"),
    ]

    for X, y, case_name in invalid_cases:
        try:
            NumpyLinearRegression().fit(X, y)
        except ValueError:
            continue
        raise AssertionError(f"{case_name} 应该抛出 ValueError。")


def test_analytical_gradient_matches_numerical_gradient() -> None:
    """
    核心测试：解析梯度应该和数值梯度接近。

    面试中如果你手写反向传播，最可靠的自检方式之一就是 gradient check：
        1. 用公式推导得到解析梯度。
        2. 用微小扰动 epsilon 近似数值梯度。
        3. 比较二者是否接近。
    """
    X = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]])
    y = np.array([1.0, 2.0, -1.0])

    model = NumpyLinearRegression(learning_rate=0.1, n_epochs=1)
    model._initialize_parameters(n_features=2)
    model.weight_ = np.array([0.4, -0.2])
    model.bias_ = 0.3

    _, grad_w, grad_b = model._compute_loss_and_gradients(X, y)

    epsilon = 1e-6
    numerical_grad_w = np.zeros_like(model.weight_)

    for idx in range(model.weight_.shape[0]):
        original_value = model.weight_[idx]

        model.weight_[idx] = original_value + epsilon
        loss_plus = model.mse_loss(model.predict(X), y)

        model.weight_[idx] = original_value - epsilon
        loss_minus = model.mse_loss(model.predict(X), y)

        numerical_grad_w[idx] = (loss_plus - loss_minus) / (2.0 * epsilon)
        model.weight_[idx] = original_value

    original_bias = model.bias_
    model.bias_ = original_bias + epsilon
    loss_plus = model.mse_loss(model.predict(X), y)

    model.bias_ = original_bias - epsilon
    loss_minus = model.mse_loss(model.predict(X), y)

    numerical_grad_b = (loss_plus - loss_minus) / (2.0 * epsilon)
    model.bias_ = original_bias

    _assert_allclose(grad_w, numerical_grad_w, atol=1e-6, message="解析 grad_w 与数值梯度不一致。")
    _assert_allclose(grad_b, numerical_grad_b, atol=1e-6, message="解析 grad_b 与数值梯度不一致。")


def run_all_tests() -> None:
    """运行所有测试用例。"""
    tests = [
        test_single_feature_recovers_parameters,
        test_multi_feature_recovers_parameters,
        test_constant_target_learns_bias,
        test_zero_epoch_only_initializes_parameters,
        test_predict_before_fit_raises_error,
        test_invalid_inputs_raise_errors,
        test_analytical_gradient_matches_numerical_gradient,
    ]

    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")


def run_demo() -> None:
    """演示一次完整训练，让你看到真实参数和学习到的参数。"""
    X, y, true_w, true_b = make_synthetic_linear_data(
        n_samples=300,
        noise_std=0.2,
        random_seed=42,
    )

    model = NumpyLinearRegression(
        learning_rate=0.05,
        n_epochs=1000,
        tol=1e-12,
        verbose=False,
    )
    model.fit(X, y)

    predictions = model.predict(X)
    final_loss = model.mse_loss(predictions, y)

    print("\n=== Demo: NumPy Linear Regression From Scratch ===")
    print(f"true_w      = {true_w}")
    print(f"learned_w   = {model.weight_}")
    print(f"true_b      = {true_b:.6f}")
    print(f"learned_b   = {model.bias_:.6f}")
    print(f"final_mse   = {final_loss:.8f}")
    print(f"epochs_run  = {model.history_.epochs_run}")
    print("first_5_predictions_vs_labels:")

    for pred, label in zip(predictions[:5], y[:5]):
        print(f"  pred={pred: .6f}, label={label: .6f}")


if __name__ == "__main__":
    run_all_tests()
    run_demo()
