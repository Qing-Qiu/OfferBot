# MkDocs 浏览层使用指南

本项目的课程正文仍然保存在根目录下的 `00_Python_Syntax` 到 `07_Algorithm` 中。`docs/` 目录只是 MkDocs 的浏览包装层，通过片段引用把原始 Markdown 渲染成文档站。

## 一次性安装

建议单独创建一个轻量文档环境：

```powershell
conda create -n offerbot-docs python=3.11
conda activate offerbot-docs
pip install -r requirements-docs.txt
```

## 本地预览

在项目根目录运行：

```powershell
conda activate offerbot-docs
python tools/serve_mkdocs.py
```

然后打开：

```text
http://127.0.0.1:8000/OfferBot/
```

`tools/serve_mkdocs.py` 默认会把 MkDocs 临时目录放到已忽略的 `site/.mkdocs_tmp/`，避免 `.mkdocs_tmp/` 出现在项目根目录干扰递归扫描。也可以用环境变量 `OFFERBOT_MKDOCS_TMP` 指定其他临时目录。

## 停止预览服务

查看 8000 端口对应的进程：

```powershell
netstat -ano | Select-String ':8000'
```

停止服务：

```powershell
Stop-Process -Id <PID>
```

## 构建静态站点

```powershell
mkdocs build
```

构建产物会生成到 `site/`，该目录已经加入 `.gitignore`，通常不需要提交。

如果 Windows 临时目录权限导致 `mkdocs serve` 失败，请优先使用上面的 `python tools/serve_mkdocs.py`，而不是直接运行 `mkdocs serve`。

## 编辑规则

- 正文内容：编辑根目录下的课程文件，例如 `01_Math_Foundations/06_matrix_derivatives/README.md`。
- 导航结构：编辑 `mkdocs.yml`。
- MkDocs 包装页：`docs/` 下的文件只保留引用语句，一般不要写正文。
