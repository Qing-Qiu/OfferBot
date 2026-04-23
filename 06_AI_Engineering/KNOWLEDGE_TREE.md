# 06 AI 工程化与系统设计知识树

目标：掌握 AI 模型从训练到上线的完整工程链路，能回答大厂算法/AI 岗常见系统设计与落地问题。

## 建议小章节

```text
01_mlops_overview
02_data_pipeline
03_feature_store
04_training_engineering
05_model_serving
06_vector_search_system
07_rag_system_design
08_recsys_system_design
09_monitoring_and_abtest
10_engineering_tools
11_system_design_interview
```

说明：这章的小章节按工程链路拆，便于后续直接沉淀系统设计讲义和实战 checklist。

## A. AI 工程化总览

- A1 离线训练、离线评估、在线服务
- A2 数据链路、特征链路、模型链路
- A3 训练任务 vs 推理服务
- A4 batch inference vs online inference
- A5 延迟、吞吐、可用性、成本
- A6 灰度、回滚、降级
- A7 监控、报警、故障定位
- A8 工程化面试答题框架

## B. 数据工程基础

- B1 日志埋点
- B2 曝光、点击、转化日志
- B3 样本构造
- B4 正负样本定义
- B5 数据清洗
- B6 数据质量校验
- B7 数据分区与时间窗口
- B8 SQL 基础：select / where / group by / join
- B9 离线表、实时流、宽表
- B10 数据泄漏与时间穿越

## C. 特征工程系统

- C1 离线特征
- C2 在线特征
- C3 实时特征
- C4 Feature Store
- C5 特征注册、版本、血缘
- C6 训练/线上一致性
- C7 特征缺失与默认值
- C8 特征延迟
- C9 特征监控
- C10 特征回填

## D. 训练工程

- D1 训练配置管理
- D2 seed 与可复现
- D3 checkpoint
- D4 断点续训
- D5 early stopping
- D6 mixed precision
- D7 gradient accumulation
- D8 gradient clipping
- D9 分布式训练：Data Parallel / DDP
- D10 显存优化：activation checkpointing / offload
- D11 训练日志与实验追踪
- D12 模型版本管理

## E. 模型评估与实验

- E1 离线指标
- E2 在线指标
- E3 训练/验证/测试切分
- E4 时间切分
- E5 交叉验证
- E6 A/B 实验
- E7 流量分桶
- E8 显著性与置信区间
- E9 灰度发布
- E10 离线涨线上不涨
- E11 指标漂移
- E12 人工评测

## F. 模型服务与推理系统

- F1 模型导出：state_dict / TorchScript / ONNX
- F2 模型加载与热更新
- F3 REST / gRPC
- F4 online inference
- F5 batch inference
- F6 异步队列
- F7 缓存
- F8 限流、熔断、降级
- F9 延迟优化
- F10 吞吐优化
- F11 CPU/GPU 资源调度
- F12 线上一致性测试

## G. 向量检索系统

- G1 embedding 生成
- G2 向量索引
- G3 Faiss
- G4 HNSW
- G5 IVF / PQ
- G6 Milvus / 向量数据库
- G7 索引构建
- G8 增量更新
- G9 召回率与延迟权衡
- G10 向量召回系统设计

## H. RAG 工程系统

- H1 文档解析
- H2 文本切分
- H3 embedding 任务
- H4 向量入库
- H5 query rewrite
- H6 多路召回
- H7 rerank
- H8 prompt 组装
- H9 引用与可追溯
- H10 RAG 评估
- H11 hallucination 监控
- H12 权限与数据隔离

## I. 推荐/广告系统设计

- I1 推荐系统整体架构
- I2 召回服务
- I3 粗排服务
- I4 精排服务
- I5 重排与规则层
- I6 特征服务
- I7 样本回流
- I8 模型更新频率
- I9 冷启动系统设计
- I10 热门偏置与多样性
- I11 广告竞价基础
- I12 搜索排序系统设计

## J. 观测、稳定性与安全

- J1 服务日志
- J2 trace
- J3 指标监控
- J4 数据漂移监控
- J5 模型效果监控
- J6 异常报警
- J7 回滚策略
- J8 权限控制
- J9 隐私与脱敏
- J10 prompt injection 防护
- J11 内容安全
- J12 审计与合规

## K. 工程工具基础

- K1 Git 基础
- K2 Linux 常用命令
- K3 Docker 基础
- K4 环境管理
- K5 配置文件
- K6 日志系统
- K7 单元测试
- K8 性能 profiling
- K9 CI/CD 基础
- K10 云服务基础概念

## L. 面试系统设计题

- L1 设计一个短视频推荐系统
- L2 设计一个电商搜索排序系统
- L3 设计一个广告 CTR 预估系统
- L4 设计一个双塔向量召回系统
- L5 设计一个 RAG 问答系统
- L6 设计一个企业知识库助手
- L7 设计一个模型训练平台
- L8 设计一个在线特征服务
- L9 设计一个 A/B 实验平台
- L10 设计一个大模型评测系统

## 优先级

```text
必学：
A
B1-B10
C1-C9
D1-D8, D11-D12
E1-E11
F1-F10
G1-G10
H1-H12
I1-I10
J1-J10
L1-L6

次学：
D9-D10
F11-F12
I11-I12
J11-J12
K
L7-L10
```
