# 环境说明

## 机器

| 项目 | 值 |
| --- | --- |
| 操作系统 | Windows |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8GB，compute capability 8.9 |
| 显卡驱动 | 595.79（报告 CUDA 13.2） |
| CUDA 编译器 | `D:\SDK\CUDA\v13.0\bin\nvcc.exe`，CUDA 13.0 |
| Python | `D:\CODER\python\anaconda\envs\ai\python.exe`，Python 3.11.15 |
| 可用磁盘 | D: 约 224GB |

## 包版本

`ai` 是共享环境；下表记录本仓库依赖的关键版本，完整快照见 `docs/ai-env-snapshot.txt`。

| 包 | 变更前 | 当前 |
| --- | --- | --- |
| torch | 2.12.0+cpu | 2.14.1+cu130 |
| torchvision | 0.27.0 | 0.29.1+cu130 |
| numpy | 2.4.6 | 2.4.6 |
| pytest | 9.0.3 | 9.0.3 |
| transformers | 5.9.0 | 5.9.0（可选，仅作现有环境记录） |

## 激活环境

```powershell
conda activate ai
python --version

# 或直接使用绝对路径
& 'D:\CODER\python\anaconda\envs\ai\python.exe' --version
```

本仓库不创建 venv，也不新建 conda 环境。原因：

- 用户明确要求复用现有 `ai` 环境，不为学习仓库做隔离；
- 所有课程、测试和 CUDA 检查在同一解释器下运行，便于排查环境问题；
- 代价是环境变更会影响其他使用 `ai` 的项目，因此每次变更都要记录并验证。

## 本仓库安装

```powershell
$py = 'D:\CODER\python\anaconda\envs\ai\python.exe'
& $py -m pip install -e . --no-deps
```

使用 `--no-deps` 是有意的：torch/torchvision 已由 CUDA 专用索引安装，不能让 pip 在解析时把它们替换成 PyPI 的 CPU 版本。

## CUDA 版 torch 安装

```powershell
$py = 'D:\CODER\python\anaconda\envs\ai\python.exe'
& $py -m pip install --upgrade `
  --index-url https://download.pytorch.org/whl/cu130 `
  "torch==2.14.1+cu130" "torchvision==0.29.1+cu130"
```

验证：

```powershell
& $py -c "import torch, torchvision; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torchvision.__version__)"
& $py -m dlexer doctor --require-gpu
```

## 回滚

如果 CUDA 版安装导致与本项目无关的依赖问题，可回滚到升级前版本：

```powershell
& $py -m pip install --force-reinstall torch==2.12.0 torchvision==0.27.0
& $py -m pip uninstall -y dlexer ruff
```

回滚后 GPU 测试会自动跳过，`dlexer doctor` 会报告 CUDA 不可用。

## 依赖一致性

安装后运行：

```powershell
& $py -m pip check
```

2026-11-10 安装 cu130 版本后的结果：`No broken requirements found.` 如果以后出现与本项目无关的既有冲突，只记录，不擅自修复。torch / torchvision 的版本关系由 PyTorch 官方索引保证；本项目使用 `torch>=2.14,<3`、`torchvision>=0.29,<1` 的宽松下界，实际值以本机快照为准。

同一次验证的关键结果：

- `torch.cuda.is_available() == True`，识别 RTX 4060 Laptop，compute capability 8.9，bf16 支持；
- `torch.backends.cudnn.version() == 92400`（9.24.0）；
- `pytest -q` 通过（GPU 用例实际运行），网络用例默认跳过；
- `dlexer doctor --require-gpu --require-network` 退出码 0，仅保留 MSVC/ninja 两条扩展工具链警告。

## 已知限制

- 自定义 CUDA 扩展需要 MSVC Build Tools + `cl.exe` 在 PATH；当前未安装，见 [cuda.md](cuda.md)。
- 8GB 显存适合小模型与课程实验；batch size 需要按实验记录调整。
- `ai` 环境中的 transformers 不是本仓库硬依赖，后续课程如使用会再单独记录。
