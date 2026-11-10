# DLexer

个人 PyTorch 从零学习工作区：用可运行、可测试的方式重写经典架构，配合 AI 教学与 agent 审查，服务于课程、科研和长期记忆。

[![checks](https://github.com/AnieerLhayK/DLexer/actions/workflows/checks.yml/badge.svg)](https://github.com/AnieerLhayK/DLexer/actions/workflows/checks.yml)

## 当前状态

| 能力 | 状态 |
| --- | --- |
| Python 环境 | `ai`（Python 3.11.15），不创建 venv、不做隔离 |
| PyTorch | `2.14.1+cu130` |
| torchvision | `0.29.1+cu130` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8GB |
| 运行时 CUDA | `dlexer doctor --require-gpu` 验证 |
| 自定义 CUDA 扩展 | 暂缓：未安装 MSVC Build Tools / ninja |
| 课程 | 00 可运行；01–06 已建立路线图和验收标准 |

## 快速开始

```powershell
conda activate ai

# 环境 / GPU / CUDA / 网络联通检查
python -m dlexer doctor --require-gpu --require-network

# 快速测试
python -m pytest -q

# 端到端训练烟测（默认自动选择 CUDA）
python -m dlexer smoke --device cuda

# 第一课
python lessons\00_foundations\lab.py
```

如果 `conda activate ai` 在你的终端不可用，直接使用本机解释器：

```powershell
& 'D:\CODER\python\anaconda\envs\ai\python.exe' -m dlexer doctor
```

## 目录

| 路径 | 用途 |
| --- | --- |
| `src/dlexer/` | 环境自检、GPU 校验、联通探测、烟测训练 |
| `tests/` | 仓库级测试：CLI、环境、GPU、训练、课程结构 |
| `lessons/` | 从张量到 CNN/RNN/Transformer 的课程与 lab |
| `docs/` | 环境、CUDA、教学/审查流程、课程映射、资料 |
| `scripts/` | 自定义 CUDA 扩展预检（当前工具链未就绪） |
| `data/` | 数据集放置位置（内容忽略） |
| `runs/` | 训练产物（内容忽略） |
| `experiments/` | 可提交的实验记录与结论 |
| `notebooks/` | 可选 notebook 说明（默认不用） |

## 学习循环

1. 读 `lessons/NN_name/README.md`，明确目标、概念与验收标准；
2. 自己写 `lab.py`，agent 只讲解和追问，不直接给成品；
3. agent 按 [docs/workflow.md](docs/workflow.md) 审查 diff，给最小修改建议；
4. 用 `pytest` 与 `notes.md` 完成验收；
5. 小步提交，更新课程状态。

## 环境与 CUDA

- 复用 conda 环境 `ai`，不创建隔离环境；详见 [docs/environment.md](docs/environment.md)。
- 运行时 CUDA 与自定义扩展链接的区别、当前工具链缺口，见 [docs/cuda.md](docs/cuda.md)。
- 重装或回滚命令都记录在环境文档中；不要在未确认的情况下把 CUDA 版 torch 换回 CPU 版。

## 远端

- GitHub：<https://github.com/AnieerLhayK/DLexer>
- 默认分支：`main`
- 公开仓库：提交前确认没有数据、密钥或个人信息。

## License

MIT，见 [LICENSE](LICENSE)。
