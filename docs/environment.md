# 环境与 VS Code

本项目使用 conda 环境 `dlexer`，位于 `D:\CODER\python\anaconda\envs\dlexer`。Python 和其他依赖独立；为节省空间，`torch`、`torchvision`、`torchgen` 和 `functorch` 通过 Windows junction 共享 `ai` 环境中的文件。两个包的版本元数据保存在 `dlexer` 中。

**维护边界：** `ai` 中的 PyTorch 更新或卸载会影响本项目，也可能让 `dlexer` 的版本元数据过时。不要在任一环境中卸载、覆盖或升级这两个共享包，直到先解除 junction 并确认后续安装方案。`pip check` 不能证明共享文件仍与元数据一致；每次 `ai` 的 PyTorch 变动后需重新验证导入、版本和 GPU。

## 在另一台机器创建独立环境

下面的命令用于**尚无 junction 的新环境**。本机已有的 `dlexer` 不应直接执行 PyTorch 安装命令，以免影响共享文件。

```powershell
conda create -n dlexer python=3.11 pip
conda activate dlexer
python -m pip install --index-url https://download.pytorch.org/whl/cu130 "torch==2.14.1+cu130" "torchvision==0.29.1+cu130"
python -m pip install -e ".[dev]"
python -m pip check
```

PyTorch wheel 自带运行所需的 CUDA 库；显卡驱动必须正常。已装的 CUDA Toolkit 可以保留，本项目不需要安装编译器、MSVC 或 ninja。版本与索引应以 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)为准。

## VS Code

用 VS Code 打开 `D:\AI\projects\DLexer` 文件夹。项目设置指定了本机 `dlexer` 解释器并启用 pytest；首次打开后在状态栏确认解释器为 `dlexer`。若没有自动选择，执行“Python: Select Interpreter”，选择 `D:\CODER\python\anaconda\envs\dlexer\python.exe`。

打开 Python 文件后按 F5，可用“运行当前 Python 文件”调试；测试面板可发现 `tests/`。终端内如果 `conda activate dlexer` 不可用，可直接使用解释器绝对路径。

## 验证

```powershell
python -m pip check
python -m dlexer doctor --require-gpu --require-network
python -m dlexer smoke --device cuda
python -m pytest -q
python -m ruff check .
python lessons\00_foundations\lab.py
```

GPU 故障排查见 [cuda.md](cuda.md)。需要修改依赖时先说明用途和影响；不使用 `ai` 的卸载或回滚命令。
