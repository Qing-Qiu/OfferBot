# 00 Python 语法专题知识树

目标：读懂 AI 项目里的 Python、NumPy、pandas、PyTorch 代码。

## 当前小章节目录

```text
01_python_core
02_python_oop
03_python_code_reading
04_numpy_basics
05_pandas_basics
06_pytorch_syntax_preview
07_debug_and_testing
```

说明：上面的小章节目录已经确定；下面的 A/B/C... 继续作为叶子知识点树，用于后续展开讲义和代码。

## A. Python 核心语法

- A1 变量、对象、引用
- A2 数字、字符串、布尔值、None
- A3 list / tuple / dict / set
- A4 if / for / while / break / continue
- A5 函数、返回值、默认参数、关键字参数
- A6 `*args` / `**kwargs`
- A7 作用域、可变对象陷阱
- A8 class / self / `__init__`
- A9 实例属性、类属性、方法
- A10 `@dataclass`
- A11 类型标注：`Optional` / `list[T]` / `tuple[...]`
- A12 异常：`try` / `except` / `raise`
- A13 模块导入：`import` / `from ... import ...`
- A14 程序入口：`if __name__ == "__main__"`

## B. Python 代码阅读

- B1 列表/字典/集合推导式
- B2 `enumerate` / `zip` / `range`
- B3 `lambda` / `sorted(key=...)`
- B4 `with` 上下文管理器
- B5 单下划线 `_method` 与尾随下划线 `weight_`
- B6 断言与简单测试
- B7 文件路径、相对路径、绝对路径
- B8 常见调试：`print(type(x), shape)`

## C. NumPy

- C1 `ndarray` / `shape` / `ndim` / `dtype`
- C2 创建数组：`array` / `zeros` / `ones` / `arange` / `linspace`
- C3 reshape / squeeze / expand_dims
- C4 索引和切片：行、列、mask
- C5 矩阵乘法：`@` / `dot` / `matmul`
- C6 广播机制
- C7 聚合：sum / mean / max / argmax
- C8 随机数：`default_rng`
- C9 数值稳定：NaN / Inf / log(0)
- C10 向量化替代循环

## D. pandas

- D1 Series / DataFrame
- D2 读取与查看：read_csv / head / info / describe
- D3 列选择、条件筛选
- D4 新增列、删除列、重命名
- D5 缺失值：isna / fillna / dropna
- D6 groupby / agg
- D7 merge / concat
- D8 sort_values / drop_duplicates
- D9 时间列处理
- D10 推荐系统特征工程常用写法

## E. PyTorch 语法预备

- E1 Tensor 创建、shape、dtype、device
- E2 Tensor 索引、切片、reshape、permute
- E3 Tensor 与 NumPy 转换
- E4 `requires_grad` / `.grad` / `backward`
- E5 `nn.Module` / `__init__` / `forward`
- E6 `Dataset` / `DataLoader`
- E7 loss / optimizer / train loop
- E8 `train()` / `eval()` / `no_grad()`
- E9 `state_dict` 保存加载

## 优先级

```text
必学：A1-A14, C1-C7, E1-E8
次学：B1-B8, D1-D8
补充：C8-C10, D9-D10, E9
```
