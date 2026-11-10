# GPU 运行与排查

本项目只通过 Python 和 PyTorch 使用 GPU。安装了支持 CUDA 的 PyTorch wheel 后，运行 GPU 代码需要可用的 NVIDIA 显卡驱动；无需安装 CUDA Toolkit 或 MSVC。既有系统工具保持原样。

```powershell
python -m dlexer doctor --require-gpu
python -m dlexer smoke --device cuda
```

- `torch.version.cuda` 为空：检查是否误装了 CPU 版本，按 [environment.md](environment.md) 安装 GPU wheel。
- GPU 不可用：运行 `nvidia-smi` 查看驱动，再检查 `torch.cuda.is_available()`。
- 显存不足：减小 batch size 或序列长度，减少同时保留的张量。
- CI 使用 CPU，GPU 验证在本机执行。
