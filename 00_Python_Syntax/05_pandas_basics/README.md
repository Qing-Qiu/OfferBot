# 05 pandas_basics

本小章节用于读懂表格数据处理与特征工程代码：

- `Series` / `DataFrame`
- 列选择与条件筛选
- `groupby`
- `merge`
- 缺失值处理

## 学习目标

pandas 在本项目里服务推荐系统和特征工程。最低目标不是背 API，而是能看懂：

```text
一列如何被筛选、变换、聚合
多张表如何 join 成训练样本
缺失值如何影响模型特征
时间窗口统计特征如何构造
```

## 后续展开顺序

1. Series / DataFrame / index。
2. 列选择、条件筛选、新增列。
3. 缺失值：`isna` / `fillna` / `dropna`。
4. `groupby` / `agg` 统计特征。
5. `merge` / `concat` 连接用户、物品、行为表。
