# 04 推荐系统知识树

目标：掌握召回、排序、特征、样本、评估和线上工程的完整链路。

## 建议小章节

```text
01_recsys_overview
02_recall
03_two_tower
04_ranking_models
05_feature_engineering
06_samples_and_losses
07_evaluation_and_abtest
08_recsys_interview
```

说明：推荐系统后续建议按“链路模块”建小章节，而不是按单个模型名无限拆目录。

## A. 推荐系统总览

- A1 业务目标：CTR / CVR / GMV / 留存 / 时长
- A2 召回、粗排、精排、重排
- A3 离线训练 vs 在线 serving
- A4 用户、物品、上下文
- A5 曝光、点击、转化、负反馈
- A6 多目标权衡

## B. 召回

- B1 热门召回
- B2 规则召回
- B3 UserCF
- B4 ItemCF
- B5 Swing
- B6 矩阵分解
- B7 向量召回
- B8 ANN：Faiss / HNSW / IVF / PQ
- B9 召回融合
- B10 召回评估

## C. 双塔模型

- C1 User Tower
- C2 Item Tower
- C3 Embedding
- C4 相似度：内积 / cosine
- C5 负采样
- C6 batch 内负样本
- C7 hard negative
- C8 线上索引构建
- C9 双塔表达能力局限

## D. 排序模型

- D1 LR
- D2 GBDT + LR
- D3 Wide & Deep
- D4 FM / FFM
- D5 DeepFM
- D6 NFM / AFM
- D7 DCN / xDeepFM
- D8 DIN / DIEN / DSIN
- D9 MMOE / PLE
- D10 序列推荐：YouTube DNN / SASRec / BERT4Rec

## E. 特征工程

- E1 稀疏特征
- E2 稠密特征
- E3 序列特征
- E4 交叉特征
- E5 统计特征
- E6 Embedding 维度
- E7 分桶、归一化、截断
- E8 特征穿越
- E9 特征泄漏
- E10 训练/线上一致性

## F. 样本与损失

- F1 曝光样本
- F2 点击样本
- F3 转化样本
- F4 正负样本构造
- F5 位置偏差
- F6 曝光偏差
- F7 Pointwise / Pairwise / Listwise
- F8 BCE / BPR / Softmax loss
- F9 多任务损失

## G. 评估与实验

- G1 AUC / LogLoss
- G2 Recall / HitRate
- G3 NDCG / MAP
- G4 CTR / CVR / GMV
- G5 A/B 实验
- G6 显著性
- G7 离线涨线上不涨
- G8 多样性、新颖性、长尾

## H. 面试重点

- H1 为什么分召回和排序
- H2 DeepFM 如何建模特征交互
- H3 双塔为什么适合召回
- H4 冷启动
- H5 热门偏置
- H6 特征一致性
- H7 样本选择偏差

## 优先级

```text
必学：A, B3-B10, C, D1-D8, E, F1-F8, G1-G7, H
次学：D9-D10, F9, G8
```
