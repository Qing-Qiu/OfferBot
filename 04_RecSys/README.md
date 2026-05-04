# 04 RecSys：推荐系统

推荐系统不是几个模型名的堆叠，而是一条完整链路：

```text
用户请求 -> 多路召回 -> 粗排 -> 精排 -> 重排 -> 展示 -> 日志回流 -> 训练更新
```

本章目标是让你能讲清楚推荐系统为什么分层、每层解决什么问题、模型和特征如何在线上线下闭环。

## 当前内容

- [01_recsys_overview](./01_recsys_overview/README.md)：召回/粗排/精排/重排、曝光点击转化、多目标、离线在线链路。
- [02_recall](./02_recall/README.md)：UserCF、ItemCF、Swing、矩阵分解、向量召回、ANN、召回评估。
- [04_ranking_models](./04_ranking_models/README.md)：LR、Wide&Deep、FM、DeepFM、DIN、DCN、排序特征和损失。

## 学习顺序

1. 先看总览，建立业务链路。
2. 再看召回，理解候选集从哪里来。
3. 最后看排序，理解为什么需要复杂模型和特征交叉。
