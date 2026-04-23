# 01 Python 类、self、__init__：读懂模型代码的第一道门

> 学习目标：看懂 `class NumpyLinearRegression`、`self.weight_`、`fit()`、`predict()` 这类机器学习代码写法。

## 1. 为什么 AI 代码里经常写 class

机器学习模型通常不是一个孤立函数，而是一组“状态 + 行为”。

状态包括：

- 学到的权重 `weight_`
- 学到的偏置 `bias_`
- 学习率 `learning_rate`
- 训练轮数 `n_epochs`
- 训练历史 `history_`

行为包括：

- 训练：`fit(X, y)`
- 预测：`predict(X)`
- 前向计算：`_forward(X)`
- 输入校验：`_validate_x(X)`
- 计算损失和梯度：`_compute_loss_and_gradients(X, y)`

所以机器学习代码经常写成：

```python
model = NumpyLinearRegression(learning_rate=0.05, n_epochs=1000)
model.fit(X, y)
pred = model.predict(X_test)
```

这比把所有变量都散落在很多函数参数里更清晰。

## 2. class 是什么

`class` 可以理解为“对象的模板”。

```python
class LinearModel:
    pass
```

这只是定义了一个模型类型，还没有真正创建模型对象。

真正创建对象要调用类名：

```python
model = LinearModel()
```

这里：

- `LinearModel` 是类。
- `model` 是对象，也叫实例。
- 一个类可以创建很多对象。

```python
model_a = LinearModel()
model_b = LinearModel()
```

`model_a` 和 `model_b` 是两个不同对象，它们可以保存不同的参数。

## 3. __init__ 是什么

`__init__` 是对象创建后自动执行的初始化方法。

```python
class LinearModel:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
```

当你写：

```python
model = LinearModel(weight=3.0, bias=2.0)
```

Python 会自动执行：

```python
model.__init__(weight=3.0, bias=2.0)
```

初始化之后，`model` 这个对象内部就保存了：

```python
model.weight == 3.0
model.bias == 2.0
```

## 4. self 是什么

`self` 表示“当前这个对象自己”。

```python
class LinearModel:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def predict_one(self, x):
        return self.weight * x + self.bias
```

当你调用：

```python
model = LinearModel(3.0, 2.0)
model.predict_one(10.0)
```

Python 实际上会把 `model` 自动传给 `self`：

```python
LinearModel.predict_one(model, 10.0)
```

所以这两行等价：

```python
model.predict_one(10.0)
LinearModel.predict_one(model, 10.0)
```

面试或读代码时记住一句话：

```text
self.xxx 就是当前对象自己的 xxx。
```

## 5. 实例属性：self.weight 和 self.bias

写在 `self` 上的变量叫实例属性。

```python
self.weight = weight
self.bias = bias
```

它们属于具体对象。

```python
model_a = LinearModel(weight=3.0, bias=2.0)
model_b = LinearModel(weight=-1.0, bias=5.0)
```

此时：

```python
model_a.weight == 3.0
model_b.weight == -1.0
```

两个模型互不影响。

这就是为什么机器学习模型可以这样保存“训练后学到的参数”：

```python
self.weight_ = ...
self.bias_ = ...
```

## 6. 为什么有些属性后面带下划线

在很多机器学习库里，训练后才会出现的属性常常用尾随下划线：

```python
self.weight_
self.bias_
self.history_
```

这个风格来自 scikit-learn 一类的机器学习 API 习惯。

它表达的意思是：

```text
这个属性不是用户传入的超参数，而是模型 fit 之后学出来的结果。
```

例如：

```python
model = NumpyLinearRegression()
model.weight_  # fit 之前没有真正学到
model.fit(X, y)
model.weight_  # fit 之后才代表学到的权重
```

## 7. 方法是什么

写在类里面的函数叫方法。

```python
class LinearModel:
    def predict_one(self, x):
        return self.weight * x + self.bias
```

方法和普通函数最大的区别：

- 普通函数需要你手动传所有东西。
- 方法可以通过 `self` 访问对象内部保存的状态。

普通函数写法：

```python
def predict_one(x, weight, bias):
    return weight * x + bias
```

类方法写法：

```python
model.predict_one(x)
```

类方法更适合模型，因为模型参数已经保存在对象内部，不需要每次都传。

## 8. fit 和 predict 为什么这样设计

机器学习里常见两步：

```python
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

`fit` 的职责：

- 接收训练数据。
- 学习参数。
- 把学到的参数保存在 `self.weight_`、`self.bias_` 等属性中。
- 返回 `self`，方便链式调用。

`predict` 的职责：

- 使用已经学到的参数。
- 对新输入做预测。
- 不应该修改模型参数。

这个分工非常重要：

```text
fit 会改变模型状态。
predict 只是使用模型状态。
```

## 9. 私有风格方法：_forward 和 _check_is_fitted

Python 中单下划线开头通常表示“内部使用”。

```python
def _forward(self, X):
    ...

def _check_is_fitted(self):
    ...
```

这不是强制私有，只是一种约定：

```text
这个方法主要给类内部调用，外部用户通常不应该直接用。
```

对比：

- `fit()`：公开 API，用户应该调用。
- `predict()`：公开 API，用户应该调用。
- `_forward()`：内部细节，用户通常不直接调用。
- `_check_is_fitted()`：内部校验，用户通常不直接调用。

## 10. staticmethod 是什么

有些函数逻辑上属于这个类，但不需要访问 `self`。

比如输入校验：

```python
@staticmethod
def _validate_x(X):
    ...
```

它不需要 `self.weight_` 或 `self.bias_`，只检查传入的 `X` 是否合法。

调用方式可以是：

```python
NumpyLinearRegression._validate_x(X)
model._validate_x(X)
```

读代码时看到 `@staticmethod`，你可以理解为：

```text
这是放在类里的普通工具函数，不依赖具体对象状态。
```

## 11. classmethod 是什么

`@classmethod` 的第一个参数通常叫 `cls`，表示“当前这个类”。

```python
@classmethod
def _validate_xy(cls, X, y):
    X = cls._validate_x(X)
    y = cls._validate_y(y)
    return X, y
```

这里用 `cls._validate_x`，意思是从当前类上调用 `_validate_x`。

你现在不用急着深入元编程，只要先记住：

```text
self 指当前对象。
cls 指当前类。
```

## 12. 对照线性回归代码

看这段：

```python
model = NumpyLinearRegression(
    learning_rate=0.05,
    n_epochs=1000,
    tol=1e-12,
    verbose=False,
)
model.fit(X, y)
predictions = model.predict(X)
```

逐行拆解：

```text
1. NumpyLinearRegression(...) 创建一个模型对象。
2. __init__ 自动保存 learning_rate、n_epochs、tol、verbose。
3. model.fit(X, y) 开始训练，并把学到的 weight_、bias_ 存到 model 自己身上。
4. model.predict(X) 使用 model 自己的 weight_、bias_ 进行预测。
```

再看：

```python
self.weight_ = np.zeros(n_features)
self.bias_ = 0.0
```

意思是：

```text
当前这个模型对象有了自己的权重和偏置。
```

再看：

```python
return X @ self.weight_ + self.bias_
```

意思是：

```text
拿输入 X 和当前模型自己的 weight_、bias_ 计算预测值。
```

## 13. 常见误区

### 误区 1：以为 self 需要手动传

不需要。

```python
model.predict(X)
```

Python 会自动把 `model` 传给 `self`。

### 误区 2：分不清类和对象

```python
NumpyLinearRegression
```

这是类。

```python
model = NumpyLinearRegression()
```

`model` 是对象。

### 误区 3：以为 __init__ 是训练

`__init__` 只是初始化超参数和空状态，不应该真正训练模型。

真正训练发生在：

```python
model.fit(X, y)
```

### 误区 4：把超参数和训练参数混在一起

超参数：

```python
self.learning_rate
self.n_epochs
```

训练学到的参数：

```python
self.weight_
self.bias_
```

### 误区 5：不检查模型是否已经 fit

如果还没训练就预测，`weight_` 和 `bias_` 还没有有效值。

所以模型代码常有：

```python
self._check_is_fitted()
```

## 14. 读类代码的固定顺序

遇到一个陌生类，按这个顺序读：

1. 看类名：它代表什么对象。
2. 看 `__init__`：对象创建时保存了哪些信息。
3. 看公开方法：比如 `fit`、`predict`、`transform`。
4. 看训练后属性：比如 `weight_`、`bias_`、`history_`。
5. 看内部方法：比如 `_forward`、`_validate_x`。
6. 最后看测试：确认这个类应该怎么被使用。

## 15. 小练习

不运行代码，先判断下面输出是什么：

```python
class ToyModel:
    def __init__(self, weight):
        self.weight = weight

    def predict(self, x):
        return x * self.weight

model_a = ToyModel(2)
model_b = ToyModel(5)

print(model_a.predict(10))
print(model_b.predict(10))
```

答案：

```text
20
50
```

原因：`model_a` 和 `model_b` 是两个不同对象，它们各自保存自己的 `weight`。

