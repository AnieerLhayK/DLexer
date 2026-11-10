# 00 · 张量、自动微分与第一次训练

## 目标

- 熟悉 tensor 的 shape、dtype、device 和广播规则；
- 理解 autograd 如何从标量 loss 反向传播；
- 用解析梯度和 autograd 对照，建立“梯度检查”的习惯；
- 用最少的代码完成一次线性回归训练，确认训练循环的各个环节。

## 运行

```powershell
python lessons\00_foundations\lab.py
```

## 任务

1. 跑通 `lab.py`，确认第 2 节的 `max |grad - analytic|` 接近 0。
2. 修改线性回归的学习率（0.01、0.1、1.0），观察 loss 曲线与发散情况。
3. 把线性回归改成两层 MLP，在 XOR 数据上训练；先不追求高准确率，只确认 loss 能下降。
4. 记录：为什么 `loss.backward()` 之前必须 `optimizer.zero_grad()`？
5. 记录：`detach()`、`no_grad()`、`requires_grad` 分别在什么时候使用？

## 验收标准

- [ ] `lab.py` 正常运行，无异常；
- [ ] 梯度检查误差小于 `1e-5`；
- [ ] 能解释 `features.T @ predictions` 为什么是权重的解析梯度；
- [ ] 在 `notes.md` 写下至少两条自己的理解或疑问。

## 参考资料

- PyTorch Autograd 教程：<https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html>
- Dive into Deep Learning 线性回归：<https://d2l.ai/chapter_linear-networks/linear-regression.html>
- 3Blue1Brown 神经网络系列：<https://www.3blue1brown.com/topics/neural-networks>
