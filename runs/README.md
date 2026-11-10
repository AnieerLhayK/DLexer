# 运行产物目录

训练产生的权重、日志、TensorBoard 事件、图片等写入 `runs/`，**内容默认不提交**（`.gitignore` 中 `runs/*`，仅保留本 README）。

建议结构：

```text
runs/2026-11-10-mlp-xor-001/
├─ config.json      # 参数、seed、命令
├─ metrics.jsonl    # 每步/每 epoch 指标
├─ best.pt          # 权重
└─ summary.md       # 一句话结论
```

只有需要在仓库中长期保留的**结论**才写入 `experiments/`，不要提交大文件。
