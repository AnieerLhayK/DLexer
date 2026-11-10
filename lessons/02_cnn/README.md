# 02 · 卷积神经网络

## 目标

- 掌握卷积输出尺寸、感受野、padding/stride/dilation 的计算；
- 手写 2D 卷积与池化，并与 `torch.nn.functional` 对照；
- 理解 BatchNorm、残差连接和全局平均池化的作用；
- 在 FashionMNIST/CIFAR-10 上完成一次可复现的 CNN 训练。

## 核心概念

- 互相关与卷积的区别，通道维与 batch 维；
- `out = floor((in + 2p - k) / s) + 1`；
- im2col：把卷积变成矩阵乘法，理解 `unfold`；
- 池化、stride、全局平均池化；
- BatchNorm 的 train/eval 行为与 running stats；
- 残差连接解决深层网络的优化问题。

## 从零实现任务

1. 用四重循环实现 naive `conv2d`，在极小张量上跑通。
2. 用 `unfold` 实现 im2col 卷积，与 `F.conv2d` 对比输出与速度。
3. 实现 `max_pool2d` 与 `BatchNorm2d` 的前向，并与 torch 对照。
4. 用 LeNet 跑 FashionMNIST，达到 90% 以上测试准确率。
5. 加一个残差块，比较 10 层与 20 层网络的训练稳定性。
6. 做一次数据增强 ablation，记录 train/val 曲线。

## 与 `torch.nn` 的对照

- 权重布局：`out_channels × in_channels × kH × kW`；
- `F.conv2d` 的 padding 模式与手写实现差异；
- BatchNorm 在 `model.eval()` 后是否还更新 running stats。

## 验收标准

- [ ] 手写卷积与 `F.conv2d` 输出 `allclose`（小尺寸、固定权重）；
- [ ] 输出尺寸公式正确，边界 padding 情况有测试；
- [ ] FashionMNIST 测试准确率 ≥90%，命令可复现；
- [ ] 记录 8GB 显存下的 batch size 与显存峰值。

## 参考资料

- Dive into Deep Learning CNN：<https://d2l.ai/chapter_convolutional-neural-networks/index.html>
- CS231n CNN：<https://cs231n.github.io/convolutional-networks/>
- PyTorch Conv2d 文档：<https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>
- ResNet 论文：<https://arxiv.org/abs/1512.03385>
