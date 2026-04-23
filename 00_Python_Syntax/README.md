# 00 Python 语法专题：读懂 AI 代码的前置地基

这个专题专门解决一个问题：为什么 AI 代码看起来不长，但每一行都同时混着 Python、NumPy、pandas、PyTorch 语法。

本目录现在已经按“小章节”整理，后续新增内容时直接落在对应小章节里，不再反复改名。

## 当前小章节结构

```text
00_Python_Syntax/
├── 01_python_core/
├── 02_python_oop/
├── 03_python_code_reading/
├── 04_numpy_basics/
├── 05_pandas_basics/
├── 06_pytorch_syntax_preview/
├── 07_debug_and_testing/
├── KNOWLEDGE_TREE.md
├── README.md
└── syntax_comparison_cheatsheet.md
```

## 各小章节定位

### 01_python_core

- 变量、对象、引用
- 基础类型与容器
- 函数、参数、返回值
- 控制流
- 类型标注、异常、入口函数

### 02_python_oop

- class、self、`__init__`
- 实例属性、类属性、方法
- `@dataclass`
- `@staticmethod`、`@classmethod`
- 机器学习代码里的 `fit` / `predict`

当前已落地：

- [01_class_self_init.md](E:/PycharmProjects/OfferBot/00_Python_Syntax/02_python_oop/01_class_self_init.md)
- [01_class_self_init_examples.py](E:/PycharmProjects/OfferBot/00_Python_Syntax/02_python_oop/01_class_self_init_examples.py)

### 03_python_code_reading

- 推导式
- `enumerate` / `zip` / `range`
- `lambda` / `sorted(key=...)`
- `with`
- 常见工程命名习惯

### 04_numpy_basics

- `ndarray`
- `shape` / `ndim` / `dtype`
- `reshape`
- 切片
- `X @ w`
- 广播
- 聚合操作

当前已落地：

- [01_core_syntax.md](E:/PycharmProjects/OfferBot/00_Python_Syntax/04_numpy_basics/01_core_syntax.md)
- [01_core_syntax_examples.py](E:/PycharmProjects/OfferBot/00_Python_Syntax/04_numpy_basics/01_core_syntax_examples.py)

### 05_pandas_basics

- `Series` / `DataFrame`
- 筛选、聚合、连接
- 缺失值处理
- 推荐系统特征工程常用写法

### 06_pytorch_syntax_preview

- Tensor
- `requires_grad`
- `nn.Module`
- `forward`
- `backward`
- `optimizer.step()`

### 07_debug_and_testing

- `assert`
- 最小可运行示例
- 单元测试意识
- shape 打印与错误定位

## 交叉速查

这份文件跨越整个 00 章，不属于任何单一小章节：

- [syntax_comparison_cheatsheet.md](E:/PycharmProjects/OfferBot/00_Python_Syntax/syntax_comparison_cheatsheet.md)

## 当前学习建议

现在最适合你的顺序是：

1. `02_python_oop/01_class_self_init.md`
2. `04_numpy_basics/01_core_syntax.md`
3. `06_pytorch_syntax_preview`

这样可以直接支撑你读懂当前的线性回归脚本和后续 PyTorch 代码。

