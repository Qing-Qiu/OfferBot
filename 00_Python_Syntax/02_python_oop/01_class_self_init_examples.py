# -*- coding: utf-8 -*-
"""
01 Python 类、self、__init__ 可运行示例。

运行方式：
    python 00_Python_Syntax/02_python_oop/01_class_self_init_examples.py

学习重点：
    1. class 是对象模板。
    2. __init__ 负责初始化对象。
    3. self 表示当前对象自己。
    4. fit 会修改模型状态，predict 使用模型状态。
    5. @staticmethod 是不依赖 self 的类内工具函数。
    6. @classmethod 可以通过 cls 访问当前类。
"""

from __future__ import annotations

from dataclasses import dataclass


class SimpleLinearModel:
    """
    一个极简线性模型。

    这个类和真正机器学习模型的区别：
        - 它不训练，weight 和 bias 是创建对象时直接给定的。
        - 它只用于解释 class、self、__init__、方法调用。
    """

    def __init__(self, weight: float, bias: float) -> None:
        # self.weight 表示“当前这个模型对象自己的 weight”。
        self.weight = weight
        self.bias = bias

    def predict_one(self, x: float) -> float:
        """对单个数字做预测：y = weight * x + bias。"""
        return self.weight * x + self.bias


class FittableMeanModel:
    """
    一个有 fit / predict 风格的小模型。

    它不是真正的线性回归，只学习训练标签 y 的平均值。
    预测时，不管输入 x 是什么，都返回训练标签均值。

    这个例子用来说明：
        - fit 会改变模型状态。
        - predict 使用 fit 后保存的状态。
        - 训练后属性常用尾随下划线，例如 mean_。
    """

    def __init__(self) -> None:
        # mean_ 是训练后才学出来的属性，所以初始设为 None。
        self.mean_: float | None = None

    def fit(self, y: list[float]) -> "FittableMeanModel":
        """学习 y 的均值，并把结果保存到 self.mean_。"""
        if len(y) == 0:
            raise ValueError("y 不能为空。")

        self.mean_ = sum(y) / len(y)
        return self

    def _check_is_fitted(self) -> None:
        """内部方法：检查模型是否已经 fit。"""
        if self.mean_ is None:
            raise RuntimeError("模型尚未训练，请先调用 fit(y)。")

    def predict(self, n: int) -> list[float]:
        """预测 n 个样本，每个样本都返回训练标签均值。"""
        if n < 0:
            raise ValueError("n 必须大于等于 0。")

        self._check_is_fitted()
        return [self.mean_] * n


class LearningRateValidator:
    """
    演示 @staticmethod。

    validate_learning_rate 逻辑上和训练配置有关，
    但它不需要访问某个具体对象的 self，所以可以写成 staticmethod。
    """

    @staticmethod
    def validate_learning_rate(learning_rate: float) -> float:
        if learning_rate <= 0:
            raise ValueError("learning_rate 必须大于 0。")
        return float(learning_rate)


@dataclass
class TrainingConfig:
    """
    演示 @dataclass 和 @classmethod。

    dataclass 会自动帮我们生成 __init__，所以不用手写：
        def __init__(self, learning_rate, n_epochs): ...
    """

    learning_rate: float
    n_epochs: int

    @classmethod
    def tiny_debug_config(cls) -> "TrainingConfig":
        """
        用 classmethod 创建一个预设配置。

        cls 表示当前类 TrainingConfig。
        cls(...) 等价于 TrainingConfig(...)，但更适合继承场景。
        """
        return cls(learning_rate=0.01, n_epochs=3)


def test_self_and_method_call_are_equivalent() -> None:
    """model.predict_one(x) 与 SimpleLinearModel.predict_one(model, x) 等价。"""
    model = SimpleLinearModel(weight=3.0, bias=2.0)

    result_a = model.predict_one(10.0)
    result_b = SimpleLinearModel.predict_one(model, 10.0)

    assert result_a == 32.0
    assert result_b == 32.0


def test_two_objects_keep_independent_state() -> None:
    """同一个类创建出的不同对象，保存各自的状态。"""
    model_a = SimpleLinearModel(weight=2.0, bias=0.0)
    model_b = SimpleLinearModel(weight=5.0, bias=1.0)

    assert model_a.predict_one(10.0) == 20.0
    assert model_b.predict_one(10.0) == 51.0


def test_fit_changes_model_state_and_predict_uses_it() -> None:
    """fit 学到 mean_，predict 使用 mean_。"""
    model = FittableMeanModel()
    model.fit([1.0, 2.0, 3.0, 4.0])

    assert model.mean_ == 2.5
    assert model.predict(3) == [2.5, 2.5, 2.5]


def test_predict_before_fit_raises_error() -> None:
    """未训练就 predict，应该报错。"""
    model = FittableMeanModel()

    try:
        model.predict(1)
    except RuntimeError:
        return

    raise AssertionError("未 fit 就 predict 应该抛出 RuntimeError。")


def test_staticmethod_and_classmethod() -> None:
    """staticmethod 不依赖 self；classmethod 用 cls 创建对象。"""
    lr = LearningRateValidator.validate_learning_rate(0.05)
    config = TrainingConfig.tiny_debug_config()

    assert lr == 0.05
    assert config.learning_rate == 0.01
    assert config.n_epochs == 3


def run_all_tests() -> None:
    tests = [
        test_self_and_method_call_are_equivalent,
        test_two_objects_keep_independent_state,
        test_fit_changes_model_state_and_predict_uses_it,
        test_predict_before_fit_raises_error,
        test_staticmethod_and_classmethod,
    ]

    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")


def run_demo() -> None:
    print("\n=== Demo: class / self / __init__ ===")

    model = SimpleLinearModel(weight=3.0, bias=2.0)
    print(f"model.weight = {model.weight}")
    print(f"model.bias   = {model.bias}")
    print(f"model.predict_one(10) = {model.predict_one(10)}")

    mean_model = FittableMeanModel()
    mean_model.fit([10.0, 20.0, 30.0])
    print(f"mean_model.mean_ = {mean_model.mean_}")
    print(f"mean_model.predict(4) = {mean_model.predict(4)}")

    config = TrainingConfig.tiny_debug_config()
    print(f"debug config = {config}")


if __name__ == "__main__":
    run_all_tests()
    run_demo()
