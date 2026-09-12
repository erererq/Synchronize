# Applied Soft Computing 排版说明

## 当前文件

- `Manuscript.tex`：保留的原单栏编辑主稿。
- `Manuscript_ASC.tex`：Applied Soft Computing / Elsevier 双栏工作稿，使用 `cas-dc` 文档类。
- `Manuscript_ASC.pdf`：已编译并检查的双栏 PDF。
- 文件名含 `_ASC` 的章节或自动生成表格：仅供双栏稿调用，不改变单栏主稿。

## 编译顺序

在 `Manuscript` 目录中依次运行：

```text
xelatex Manuscript_ASC.tex
bibtex Manuscript_ASC
xelatex Manuscript_ASC.tex
xelatex Manuscript_ASC.tex
```

## 投稿时需要注意

1. Elsevier 的 LaTeX 体系同时提供单栏和双栏模板；本双栏稿采用官方 `cas-dc` 类，适合检查接近正式出版版式时的图表宽度和篇幅。
2. 初次投稿是否强制双栏、是否要求匿名稿、Highlights 或 Graphical Abstract，应以 Applied Soft Computing 投稿系统当日显示的清单为准；不要仅凭旧模板或旧版作者指南判断。
3. Editorial Manager 上传 LaTeX 源文件时通常不保留子目录结构。正式打包前，应把 `.tex`、`.bib`、类文件和所有图片依赖整理到同一级目录，并重新试编译一次。
4. 当前稿件仍带作者和单位信息，因此不能直接视为匿名审稿稿。如果系统要求匿名，应另存匿名版本，不要覆盖当前文件。

## 本次双栏化处理

- 主文档由 `cas-sc` 切换为 `cas-dc`。
- 宽图和宽表改为跨双栏浮动体。
- 普通图表保持单栏，避免字号被无谓缩小。
- 对少量在窄栏中容易越界的公式和句子进行了排版性换行，不改变实验数据或论文结论。
