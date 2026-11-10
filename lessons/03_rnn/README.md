# 03 · 循环神经网络

## 目标

- 理解循环单元、隐藏状态和按时间展开的计算图；
- 手写 RNN cell 与 BPTT，观察梯度消失/爆炸；
- 实现 LSTM/GRU 的门控机制，并与 `nn.RNN`/`nn.LSTM` 对照；
- 在序列分类或字符级任务上完成一次训练。

## 核心概念

- `h_t = tanh(x_t W_x + h_{t-1} W_h + b)`；
- BPTT：沿时间维反向传播，序列长度影响内存与梯度；
- 梯度裁剪 `clip_grad_norm_`；
- LSTM 的输入门、遗忘门、输出门与 cell state；
- GRU 的更新门、重置门；
- teacher forcing 与自回归生成的差异。

## 从零实现任务

1. 手写单层 RNN cell，逐时间步前向，输出与 `nn.RNN` 对照。
2. 手写一个序列分类训练循环，在 toy 任务（奇偶校验/序列求和）上训练。
3. 实现 LSTM cell 与 GRU cell 的前向，逐门打印激活值。
4. 构造长序列实验，比较有无梯度裁剪时的梯度范数。
5. 用字符级语言模型生成一小段文本，记录温度参数的影响。

## 与 `torch.nn` 的对照

- `nn.RNN` 的输入/输出形状与 `batch_first`；
- 初始隐藏状态、层数、dropout 的行为；
- `nn.LSTM` 返回的 `(output, (h_n, c_n))` 结构。

## 验收标准

- [ ] 手写 RNN 与 `nn.RNN` 在相同权重下输出接近；
- [ ] 梯度检查通过（短序列、小 hidden size）；
- [ ] 序列任务准确率显著高于随机；
- [ ] 记录一次梯度爆炸/消失的现象与处理。

## 参考资料

- Dive into Deep Learning RNN：<https://d2l.ai/chapter_recurrent-neural-networks/index.html>
- PyTorch RNN 文档：<https://pytorch.org/docs/stable/generated/torch.nn.RNN.html>
- LSTM 原始论文：<https://www.bioinf.jku.at/publications/older/2604.pdf>
- Karpathy RNN 有效性：<http://karpathy.github.io/2015/05/21/rnn-effectiveness/>
