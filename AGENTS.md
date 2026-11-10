# DLexer 项目规则

本仓库是登记于 Workspace 的独立个人 Python/PyTorch 架构学习项目。Workspace 不是本仓库源码目录；通过 Workspace 执行写入时遵守相应 TASK 和授权规则。先检查 Git 状态，保留用户已有修改。

## 环境

1. 使用 conda 环境 `dlexer`：`D:\CODER\python\anaconda\envs\dlexer\python.exe`。Python 和其他依赖独立；PyTorch 文件通过 junction 共享自 `ai`，见 `docs/environment.md`。
2. 依赖变更先说明目的和影响；不要在 `ai` 或 `dlexer` 中卸载、覆盖或升级共享的 PyTorch，直到解除 junction 并确认安装方案。
3. 课程实现使用 Python 和 PyTorch；GPU 通过 PyTorch 使用，不编写 C++ 或 CUDA 扩展。
4. 常用命令：
   - `python -m pytest -q`
   - `python -m ruff check .`
   - `python -m dlexer doctor --require-gpu --require-network`
   - `python -m dlexer smoke --device cuda`

## 教学与审查

1. 学习者熟悉 C++，Python 是初学者；直接学习架构，遇到 Python 语法时解释。先讲解概念、公式、形状和失败模式，再让学习者自己写 `lab.py`。默认先提示，用户明确要求时才给代码片段或修改。
2. 审查顺序：正确性 → 形状/dtype/device → 梯度 → 数值稳定性 → 性能 → 测试与文档。
3. 审查先说明问题、原因和提示；用户要求时再给最小 patch。保留学习者代码风格，不趁机重写整个文件。
4. 每一课以 `lessons/NN_name/README.md` 的验收标准为准；测试是完成门槛，不通过不宣布完成。
5. 对照实现（`reference.py`）只在 lab 完成且用户主动要求时添加，避免提前泄题。

## 数据与产物

1. 不提交数据集、checkpoint、`runs/` 产物、notebook 输出、密钥或个人信息。
2. 优先使用合成数据和小样本；真实数据下载到 `data/`，该目录内容被忽略。
3. 重要实验结果写入 `experiments/` 的 Markdown，并注明复现命令。

## Git

1. 默认分支 `main`，远端为公开仓库；提交前运行 `ruff`、`pytest` 和必要的 doctor 检查。
2. 小步提交，提交信息用英文；不 force push，不改写已推送历史。
3. 公开仓库：提交前检查是否包含本地敏感路径、凭据或未授权数据。

## 报告

如实报告验证过的命令与结果；未验证项、失败项和已知限制必须单独列出。不要用“应该可以”代替运行结果。
