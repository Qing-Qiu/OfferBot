# 07 debug_and_testing

本小章节用于建立最小调试与测试能力：

- `assert`
- shape 检查
- 最小可运行示例
- 错误定位思路
- 简单测试函数

## 学习目标

本节目标是让你能独立定位 80% 的初学代码问题：

```text
会打印 type / shape / dtype / device
会用 assert 固定关键假设
会写最小可运行示例复现错误
会把普通 case、边界 case、异常 case 分开测
```

## 后续展开顺序

1. `print(type(x), shape)` 的定位套路。
2. `assert` 与错误信息。
3. 最小可运行示例。
4. 简单测试函数组织方式。
5. 常见错误：shape mismatch、None、KeyError、device mismatch。
