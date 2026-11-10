# 04 · Transformer

## 目标

- 从零实现 scaled dot-product attention 与多头注意力；
- 理解 mask、位置编码、残差与 LayerNorm 的作用；
- 搭建一个 tiny decoder-only Transformer，完成字符级语言建模；
- 能对照 `nn.MultiheadAttention` 检查自己的实现。

## 核心概念

- `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`；
- causal mask 与 padding mask 的区别；
- 多头拆分/合并、`head_dim = d_model / num_heads`；
- 正弦位置编码与可学习位置嵌入；
- Pre-LN vs Post-LN、残差路径；
- 自回归训练中的 next-token 预测与 teacher forcing。

## 从零实现任务

1. 手写单头 attention，验证 attention 权重每行和为 1。
2. 加 causal mask，验证位置 `t` 不能看到 `t+1`。
3. 实现多头 attention，与 `nn.MultiheadAttention` 在 eval 模式下对照。
4. 实现 sinusoidal positional encoding，并可视化。
5. 拼出 Transformer block 与 tiny GPT，在字符级数据上训练到 loss 明显下降。
6. 用不同温度采样生成文本，记录重复与多样性。

## 与 `torch.nn` 的对照

- `nn.MultiheadAttention` 默认的 `batch_first` 与权重布局；
- `F.scaled_dot_product_attention` 的 mask 语义；
- `nn.LayerNorm` 的 normalized shape；
- 训练与生成时 mask 的差异。

## 验收标准

- [ ] attention 权重形状正确且每行和为 1；
- [ ] causal mask 测试通过；
- [ ] tiny GPT 在训练集上 loss 明显下降，能生成非空文本；
- [ ] 记录一次 mask/维度错误及其定位过程。

## 参考资料

- The Annotated Transformer：<https://nlp.seas.harvard.edu/annotated-transformer/>
- Attention Is All You Need：<https://arxiv.org/abs/1706.03762>
- Dive into Deep Learning Attention：<https://d2l.ai/chapter_attention-mechanisms/index.html>
- PyTorch MultiheadAttention：<https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html>
- Karpathy nanoGPT：<https://github.com/karpathy/nanoGPT>
