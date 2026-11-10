# Notebook（可选）

默认不使用 notebook：课程以 Python 脚本 + pytest 为主，便于 agent 审查、diff 和自动验收。

如果确实需要交互式讲解，可以：

1. 在 `ai` 环境注册 kernel：

```powershell
& 'D:\CODER\python\anaconda\envs\ai\python.exe' -m ipykernel install `
  --user --name dlexer-ai --display-name "Python (ai)"
```

2. 用 VS Code 选择 `Python (ai)` kernel，或使用已安装的 `jupytext` 把 `.py`（percent 格式）与 `.ipynb` 配对：

```powershell
jupytext --sync notebooks\demo.py
```

约定：

- notebook 中的关键结论要回写到 `notes.md` 或 `lessons/.../README.md`；
- 不提交带输出的 `.ipynb`（至少清空输出）；
- 不在 notebook 中保存数据路径、密钥或个人信息。
