## 修复项
- 消除启动提示：将 `translate_lmstudio.py` 的 `openai` 顶层导入改为懒加载（在 `run_translation()` 内部），GUI 导入时不打印、不退出；仅在真正开始翻译时检测依赖。
- 默认主题与切换：
  - 在 App 初始化阶段创建 `ttkbootstrap.Style`，从 `gui_prefs.json` 读取 `theme`（缺省 `minty`）并立即应用。
  - 主题下拉框创建后主动调用一次应用方法；`_on_theme_change` 复用同一 `Style` 实例执行 `theme_use`，确保打包后可切换。
  - 打包模式读取配置路径：`sys.frozen` 时从 EXE 同目录的 `gui_prefs.json` 读取。
- PDF 面板对齐：保持“识别方式 + VLM URL”“VLM Key + DPI”“页范围”三行、统一左右对齐。

## 打包与产物
- 依赖安装（uv）：`uv pip install -r requirements.txt`
- 打包命令：
  - `uvx pyinstaller --noconfirm --clean --onedir --collect-submodules ttkbootstrap --collect-data ttkbootstrap --name LM-Translate-GUI gui.py`
- 清理中间文件：删除 `build/`
- 产物：`dist/LM-Translate-GUI/LM-Translate-GUI.exe`，同目录放置 `gui_prefs.json`（含 `theme: minty` 与 `pdf: { mode, vlm_url, vlm_key, dpi, pages }`）

## 验证
1. 启动 EXE：不出现 `openai` 提示；默认主题 `minty` 已应用。
2. 下拉切换主题：立即生效无错误。
3. PDF 面板字段位置对齐；VLM 配置填写/留空均正常；DPI 200/300 两种值验证识别速度与质量差异。

## Git 提交与推送
- 提交代码与配置（不包含中间 `build/`）：
  - `git add translate_lmstudio.py gui.py document_loader.py pdf_recognizer.py requirements.txt dist/LM-Translate-GUI/gui_prefs.json`
  - `git commit -m "fix: lazy-load openai; default & switch theme (ttkbootstrap), align PDF panel; package config in dist"`
- 推送到 GitHub：
  - 设置远程（如未设置）：`git remote add origin https://github.com/<your-username>/<repo>.git`
  - 推送：`git push origin main`
  - 推送标签（可选发布版本）：`git tag v1.3.0 && git push origin v1.3.0`
- 如需创建 GitHub Release（可选）：使用 GitHub Web 或 `gh release create v1.3.0 dist/LM-Translate-GUI/LM-Translate-GUI.exe --notes "Theme fix & DPI options"`

## 说明（DPI）
- DPI（每英寸点数）用于 PDF 渲染到位图；实现中以 `fitz.Matrix(dpi/72, dpi/72)` 缩放。
- 默认 200；150–200 为速度与清晰度平衡，300 更清晰但更慢、更耗内存。

确认后我将按上述步骤修改代码、重打包生成 EXE、并执行 Git 提交与推送。