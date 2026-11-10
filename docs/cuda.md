# CUDA：运行时链接与扩展链接

本仓库把“CUDA 能用”分成两层，避免把运行时检查和自定义 kernel 编译混为一谈。

## 第一层：运行时 CUDA（当前目标）

torch 的 CUDA wheel 自带 CUDA runtime 与 cuDNN，通过显卡驱动使用 GPU。只要满足以下条件即可：

```powershell
python -m dlexer doctor --require-gpu --require-network
python -m pytest -q
python -m dlexer smoke --device cuda
```

检查项：

- `torch.version.cuda` 不是 `None`（当前目标：`13.0`）；
- `torch.cuda.is_available()` 为 `True`；
- 能识别 RTX 4060 Laptop 8GB；
- GPU 上的 512×512 fp32 matmul 与 CPU 结果接近；
- cuDNN 版本可读，bf16 支持可读。

本级不要求安装完整 CUDA Toolkit，也不要求 MSVC。

## 第二层：自定义 CUDA 扩展（暂缓）

使用 `torch.utils.cpp_extension` 编译自定义 kernel 需要：

1. CUDA 版 torch（含头文件与库）；
2. `nvcc`（本机 `D:\SDK\CUDA\v13.0`，已具备）；
3. MSVC Build Tools，包含 “使用 C++ 的桌面开发” 工作负载，`cl.exe` 能在 Developer PowerShell 中运行；
4. 建议安装 `ninja`（`pip install ninja`）。

当前缺口：未安装 MSVC，`cl.exe` 不在 PATH，`ninja` 也未安装。因此：

```powershell
python scripts\check_cuda_extension.py
```

预期以退出码 `2` 结束，并列出缺口；它不会尝试编译。工具链就绪后，同一脚本会编译并运行一个最小 `vecadd` kernel（源码在 `scripts/csrc/`）。

安装 MSVC 的参考步骤（未在本仓库执行）：

1. 安装 Visual Studio 2022 Build Tools；
2. 勾选 “使用 C++ 的桌面开发”；
3. 在 “Developer PowerShell for VS 2022” 中运行 `cl`、`nvcc` 和 `python scripts\check_cuda_extension.py`；
4. 可选：`python -m pip install ninja`。

## 故障排查

| 现象 | 检查 |
| --- | --- |
| `torch.version.cuda is None` | 装到了 CPU wheel；按 environment.md 重新安装 cu130 版本 |
| `torch.cuda.is_available() == False` | 先看 `nvidia-smi`；确认驱动正常、没有远程桌面/虚拟化限制；重启后重试 |
| `CUDA out of memory` | 减小 batch/序列长度；用 `doctor` 看显存；及时 `del` 中间张量 |
| `cl.exe not found` | 在 Developer PowerShell 中运行，或安装 VS Build Tools |
| `nvcc` 与 torch 版本不匹配 | 运行时由 wheel 自带，不依赖 nvcc；只有扩展编译需要 nvcc 与 MSVC 匹配 |
| 训练速度异常慢 | 确认张量在 `cuda`、模型在 `cuda`、没有每步 `.cpu()` 同步；用 profiler 定位 |

## 注意

- 不要把 CUDA 版 torch 换回 CPU 版来“解决”依赖问题；这会直接让本仓库的 GPU 验收失效。
- 运行 `doctor` 的 `--require-gpu` 是本地验收命令；CI 使用 CPU 版 torch，GPU 测试自动跳过。
