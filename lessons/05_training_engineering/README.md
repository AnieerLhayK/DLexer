# 05 · 训练工程

## 目标

- 掌握优化器、学习率调度、混合精度、梯度累积和 checkpoint；
- 能定位“loss 不降/NaN/显存不足/结果不可复现”的常见原因；
- 建立一套可复用的训练配置与日志习惯。

## 核心概念

- SGD vs Adam/AdamW，weight decay 的正确使用位置；
- warmup + cosine/step scheduler；
- AMP（`torch.autocast` + `GradScaler`）与 bf16/fp16 的差异；
- 梯度累积与有效 batch size；
- `torch.profiler` 与显存快照；
- 随机性来源：初始化、数据顺序、cuDNN、并行归约。

## 任务

1. 在同一模型上比较 SGD、Adam、AdamW 的收敛曲线。
2. 加 warmup + cosine，记录最佳超参。
3. 在 RTX 4060 上比较 fp32/fp16/bf16 的速度与数值稳定性。
4. 实现 checkpoint 保存/恢复，验证恢复后继续训练 loss 连续。
5. 用 profiler 找出一个训练 step 的 top-3 耗时操作。
6. 写一份“训练故障排查清单”。

## 验收标准

- [ ] 三种优化器的对比图/表；
- [ ] AMP 训练不出现 NaN，速度有记录；
- [ ] checkpoint 恢复后结果可复现（在允许误差内）；
- [ ] 至少定位并记录一次真实的性能瓶颈。

## 参考资料

- PyTorch AMP 示例：<https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html>
- PyTorch 性能调优：<https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>
- AdamW 论文：<https://arxiv.org/abs/1711.05101>
- PyTorch Profiler：<https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>
