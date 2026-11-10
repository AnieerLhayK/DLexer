# 教学、编码与审查流程

默认由学习者拥有代码，agent 负责讲解、提问、审查和补测试。学习者熟悉 C++、初学 Python：直接从架构入手，按需解释 Python 语法。默认先给提示，学习者明确要求后再提供代码片段或直接修改。

## 角色

| 角色 | 职责 | 边界 |
| --- | --- | --- |
| 学习者 | 决定目标、写 `lab.py`、记录笔记、最终提交 | 对代码负责 |
| AI 教师 | 讲概念、公式、形状、常见坑；用问题引导 | 默认不给完整实现 |
| 审查 agent | 读 diff、指出问题、给提示、补验收测试 | 用户要求时才给 patch；保留学习者的写法 |

## 一次课程的循环

### 1. 启动

agent 先读：

- `AGENTS.md`、`CONTEXT.md`、`lessons/README.md`；
- 当前课的 `README.md`；
- 相关的 `src/dlexer` 或既有实验记录（如有）。

然后讲解问题背景、核心公式、张量形状、与 `torch.nn` 的对应关系、至少一个常见错误，以及该课用到的新 Python 语法。不要直接贴完整代码。

### 2. 编码

学习者在 `lab.py` 或课程模块中实现，小步运行。遇到错误时先自己定位；需要帮助时提供最小复现。

### 3. 验收

按 `README.md` 的验收标准补测试，例如：

- 输出形状、dtype、device；
- 与 `torch.nn` 对照；
- 数值梯度检查；
- 小批量过拟合；
- 固定 seed 的复现实验。

运行：

```powershell
python -m pytest lessons\01_mlp -q
```

### 4. 审查

学习者提供 `git diff` 或未提交文件。审查 agent 按顺序检查：

1. 正确性：公式、索引、mask、loss；
2. 形状：batch/通道/时间维是否一致；
3. 梯度：`requires_grad`、`detach`、`zero_grad`、梯度裁剪；
4. 数值：softmax/log 的稳定性、dtype、除零；
5. 设备：`.to(device)`、CPU/GPU 混用、同步点；
6. 训练行为：`train()`/`eval()`、BatchNorm running stats、dropout；
7. 性能：不必要的循环、`.item()` 同步、内存复制；
8. 测试与文档：验收是否覆盖、notes 是否记录。

输出格式建议：

```text
[blocker] ... 为什么错、如何验证
[should]  ... 影响正确性或可维护性
[nice]    ... 风格/性能建议
下一步提示：...
```

### 5. 复盘

在 `notes.md` 写下：

- 这课的核心公式/直觉；
- 我原来以为什么、实际是什么；
- 至少一个失败实验和修正；
- 下一步想验证的问题。

### 6. 提交

```powershell
python -m ruff check .
python -m pytest -q
git add lessons\01_mlp
git commit -m "feat(lesson): implement mlp training loop"
```

不要提交数据、checkpoint、`runs/` 产物或 notebook 输出。

## 提示词模板

### 讲解

```text
请按 lessons/02_cnn/README.md 给我讲这一课。
先讲清卷积输出尺寸、通道布局和 im2col，再用问题和形状推导演示；
不要直接给完整实现，最后给我 3 个自查问题。
```

### 审查

```text
请审查我未提交的 diff（git diff）。
按正确性、形状、梯度、数值、设备、训练行为、性能、测试的顺序；
先列问题、原因和提示；我要求时再给最小 patch。不要重写整个文件。
```

### 调试

```text
这是我遇到的最小失败用例：<代码/报错>。
请先只读复现并定位最小原因，给我一个可以自己验证的实验；
不要直接改代码，除非我确认。
```

## 完成定义

- 验收测试通过；
- 关键结论能用一句话解释；
- `notes.md` 有记录；
- 没有未提交的产物或数据；
- 课程状态表已更新。
