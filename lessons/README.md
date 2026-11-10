# 课程地图

DLexer 的课程按“先能跑、再手写、最后对照”的顺序组织。每一课都是一个独立目录，默认由学习者自己完成 `lab.py`，agent 负责讲解、审查和补测试，而不是直接给出成品。

## 课程表

| 阶段 | 主题 | 核心产出 | 状态 |
| --- | --- | --- | --- |
| 00 | 张量、自动微分、第一次训练 | 可运行的 `lab.py` + 梯度检查 | 可运行 |
| 01 | MLP、损失函数、训练循环 | 手写 Linear/MLP + XOR 实验 | 待开始 |
| 02 | CNN、卷积算术、BatchNorm、残差 | 手写 conv/pool + FashionMNIST | 待开始 |
| 03 | RNN、BPTT、LSTM/GRU | 手写循环单元 + 序列任务 | 待开始 |
| 04 | Transformer、Attention、位置编码 | 手写 MHA + tiny GPT | 待开始 |
| 05 | 训练工程 | 优化器/调度器/AMP/可复现性 | 待开始 |
| 06 | 课程与科研专题 | 可复现实验与报告 | 待开始 |

## 每课目录契约

```text
lessons/NN_name/
├─ README.md       # 目标、概念、任务、验收标准、参考
├─ lab.py          # 学习者实现（默认不提供答案）
├─ test_lab.py     # 开始该课时由 agent 按验收标准补充
├─ notes.md        # 学习者复盘：公式、坑、对照结果
└─ reference.py    # 可选：完成 lab 后再对照的参考实现
```

- 默认 `pytest` 只收集 `tests/`；课程测试在对应目录内运行，例如：
  `python -m pytest lessons/01_mlp -q`。
- `reference.py` 只在 lab 完成并且学习者主动要求时添加，避免提前看到答案。
- 数据、checkpoint、运行产物不进入 Git。
- 每完成一课，更新本页状态，并在 `notes.md` 写下“我原来以为什么、实际是什么”。

## 验收门（Gate）

每一课至少满足：

1. 形状、dtype、device 正确；
2. 关键算子有数值梯度检查，或与 `torch.nn` 结果对照；
3. 能在小数据上过拟合到接近 100%，证明实现和训练循环没有明显错误；
4. 有可复现的 seed 与运行命令；
5. `notes.md` 记录了至少一个“踩坑/反直觉”点。

## 教学与审查流程

见 [docs/workflow.md](../docs/workflow.md)。默认节奏：agent 讲概念和公式 → 学习者写代码 → agent 只读审查并给最小修改建议 → pytest 通过 → 提交。
