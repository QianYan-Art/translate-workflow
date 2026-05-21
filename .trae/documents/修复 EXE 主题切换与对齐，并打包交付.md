## 问题与目标
- EXE 中主题切换无效：确保打包包含 ttkbootstrap 的资源与模块，并在启动时应用默认主题。
- PDF 识别选项布局对齐：同一行左中右对齐，字段位置一致。
- 打包交付：生成 dist 产物，清理中间文件；EXE 启动读取 dist/gui_prefs.json，且包含本次新增配置项（如 vlm_url 等）。

## 修复方案
- 打包修复：使用 `--collect-all ttkbootstrap` 将 ttkbootstrap 的主题资源和模块打入 EXE，避免切换主题时资源缺失。
- 默认主题应用：初始化时读取 `gui_prefs.json` 的 `theme`（默认 minty），创建 `tb.Style` 并立即应用；主题选择器构建后主动调用一次应用方法。
- PDF 面板对齐：将“识别方式 + VLM URL”放同一行，“VLM Key + DPI”同一行，“页范围”独立一行，确保左右对齐。

## 打包与配置
- 依赖安装：使用 `uv pip install -r requirements.txt`。
- 打包命令：`uvx pyinstaller --noconfirm --clean --onedir --collect-all ttkbootstrap --name LM-Translate-GUI gui.py`。
- 清理：删除 `build/`；确保 EXE 位于 `dist/LM-Translate-GUI/LM-Translate-GUI.exe`。
- 配置：同步生成/更新 `dist/LM-Translate-GUI/gui_prefs.json`，包含 `theme= minty` 与 `pdf: { mode, vlm_url, vlm_key, dpi, pages }`；打包 EXE 在启动时读取该文件。

## 验证步骤
- 启动 EXE：默认主题为 minty，无需手动切换即可生效。
- 切换主题：下拉切换其它主题应立即生效。
- 选择 PDF：对齐字段显示，填写/留空 VLM 配置分别工作；输出写入指定文件。

## 交付
- dist 目录下最终 EXE 与 gui_prefs.json；已清理中间文件。