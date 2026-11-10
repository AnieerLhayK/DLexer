# 01 · 多层感知机与训练循环

## 目标

- 理解线性层、激活函数、交叉熵和反向传播；
- 不依赖 `torch.nn.Linear` 写出前向与参数更新；
- 建立训练/验证集、过拟合小批量、正则化的基本直觉。

## 核心概念

- 线性层：`y = x @ W.T + b`，注意权重形状与 batch 维；
- 激活函数：ReLU/Tanh 的作用是引入非线性；
- logits → softmax → cross entropy 的数值稳定写法；
- 反向传播是链式法则在计算图上的自动应用；
- 过拟合、weight decay、dropout、early stopping。

## 从零实现任务

1. 用 `torch.zeros`/`nn.Parameter` 手写一个 `Linear`，前向、参数量、初始化方式与 `nn.Linear` 对照。
2. 只用 autograd 写双层 MLP，在课程的 XOR 数据上训练到 95% 以上。
3. 用 `nn.Module` 重写同一个模型，比较参数初始化、梯度范数和收敛速度。
4. 加入 weight decay 和 dropout，比较 train/val gap。
5. 用三个不同 seed 重复实验，记录均值和方差。

## 与 `torch.nn` 的对照

- 手写 `Linear` 的 `weight` 形状、初始化范围是否一致；
- 手写 softmax + NLL 与 `CrossEntropyLoss` 的数值差异；
- `optimizer.zero_grad()` 放在 backward 前后的区别。

## 验收标准

- [ ] 手写 Linear 与 `nn.Linear` 在相同输入/参数下输出接近；
- [ ] 小批量过拟合：64 个样本训练到接近 100%；
- [ ] 能解释一次 loss 不下降时排查了哪些量（学习率、初始化、标签、梯度）；
- [ ] `notes.md` 记录至少一次失败实验与修正。

## 参考资料

- Dive into Deep Learning MLP：<https://d2l.ai/chapter_multilayer-perceptrons/index.html>
- PyTorch 训练循环：<https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html>
- CS231n 反向传播：<https://cs231n.github.io/optimization-2/>
