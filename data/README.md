# 数据目录

数据集、缓存和预处理结果放在这里，**内容默认不提交**（`.gitignore` 中 `data/*`，仅保留本 README）。

约定：

- 原始数据只从用户明确指定的来源下载；
- 记录来源、版本、校验和与划分方式到对应的 `experiments/` 或 `notes.md`；
- 不把个人数据、受版权限制的数据或未脱敏数据提交到公开仓库；
- 课程优先使用 torchvision 自带数据集或合成数据，下载路径指向 `data/`。

示例：

```powershell
python -c "from torchvision.datasets import FashionMNIST; FashionMNIST(root='data', download=True)"
```
