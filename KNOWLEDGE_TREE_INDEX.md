# OfferBot 知识树索引

目标：用最少上下文保存最大结构信息。根目录只放总览；学习某一块时，只打开对应目录的 `KNOWLEDGE_TREE.md`。

## 低 token 工作约定

- 优先读索引，再读单个章节树。
- 每次只展开一个叶子知识点。
- 长讲义默认沉淀在对应目录的 `README.md`。
- 可运行代码用 `.py` 文件沉淀，必要时配套测试。
- 面经追问、quiz、review 只有内容变多时再拆成独立文件。
- 已掌握内容只保留标题和链接，不重复解释。
- 代码讲解采用“关键行 + shape/复杂度 + 面试问法”三段式。

## 大章节顺序

```text
00_Python_Syntax
  -> 读懂代码语法：Python / NumPy / pandas / PyTorch

01_Math_Foundations
  -> AI 数学地基：线性代数 / 微积分 / 概率统计 / 信息论 / 优化

02_ML_Foundations
  -> 机器学习地基：线性模型 / loss / 梯度下降 / 泛化

03_PyTorch
  -> 深度学习实现：Tensor / autograd / nn.Module / 训练循环

04_RecSys
  -> 推荐系统：召回 / 排序 / 特征 / 样本 / 评估 / 工程

05_LLM
  -> 大模型：Transformer / 训练 / 推理 / RAG / Agent / 安全

06_AI_Engineering
  -> AI 工程化：MLOps / 模型服务 / 数据链路 / 系统设计

07_Algorithm
  -> 算法面试：数据结构 / 高频范式 / DP / 图论 / 高阶题
```

## 章节知识树入口

- [00 Python 语法专题](00_Python_Syntax/KNOWLEDGE_TREE.md)
- [01 数学基础](01_Math_Foundations/KNOWLEDGE_TREE.md)
- [02 机器学习基础](02_ML_Foundations/KNOWLEDGE_TREE.md)
- [03 PyTorch 与深度学习](03_PyTorch/KNOWLEDGE_TREE.md)
- [04 推荐系统](04_RecSys/KNOWLEDGE_TREE.md)
- [05 大模型](05_LLM/KNOWLEDGE_TREE.md)
- [06 AI 工程化与系统设计](06_AI_Engineering/KNOWLEDGE_TREE.md)
- [07 算法与数据结构](07_Algorithm/KNOWLEDGE_TREE.md)

## 标准学习产物

```text
README.md       # 主讲义：原理、公式、shape、面试陷阱、自测题
demo.py         # 可运行代码：中文注释、测试、边界情况
interview.md    # 可选：面经追问、答题模板
review.md       # 可选：你的代码 review 记录
quiz.md         # 可选：自测题扩展
```
