## 启动提示消除
- 原因：`translate_lmstudio.py` 顶层导入 `openai` 失败时直接打印并 `sys.exit(1)`，GUI 导入时也会触发。
- 修复：将 `openai` 的导入移入执行路径（如 `run_translation()` 内部）并做懒加载；在 GUI 模式（`sys.frozen` 或模块被导入）不打印、不退出，仅在真正调用翻译时才校验依赖。
- 具体改动：
  1) 删除文件顶部 `try/except ImportError` 的打印与 `sys.exit(1)`。
  2) 在 `run_translation()` 开始处执行 `from openai import OpenAI` 并在缺失时抛出异常，异常由 CLI 路径提示，GUI 不在导入时出提示。

## DPI 说明
- 定义：将 PDF 页面渲染为图像的分辨率（每英寸点数），实现中通过 `fitz.Matrix(dpi/72, dpi/72)` 缩放。
- 默认：200。建议范围：
  - 150–200：速度与清晰度平衡，适合大多数文档
  - 300：更清晰但更慢、更占内存，适合小字体或扫描件
- 影响：DPI 越大图像越清晰、识别更稳，但耗时和内存开销更高。

## EXE 主题切换修复
- 代码侧：
  - 在 App 初始化时优先创建并应用 `ttkbootstrap.Style`，主题取自 `gui_prefs.json`（缺省 `minty`）。
  - 构建主题下拉后主动调用一次应用方法；`_on_theme_change` 复用同一个 `Style` 实例进行 `theme_use` 切换。
  - 打包模式下读取配置路径调整：`sys.frozen` 时从 EXE 目录读取 `gui_prefs.json`。
- 打包侧：
  - 显式收集 `ttkbootstrap` 的模块与主题资源，避免切换失败：
    - 打包命令改为：`uvx pyinstaller --noconfirm --clean --onedir --collect-submodules ttkbootstrap --collect-data ttkbootstrap --name LM-Translate-GUI gui.py`
  - 若仍有资源缺失警告：新增 `hook-ttkbootstrap.py`，内容使用 `collect_submodules('ttkbootstrap')` 与 `collect_data_files('ttkbootstrap')`，确保主题资源随 EXE 打入。

## 打包与配置
- 依赖：统一使用 `uv pip install -r requirements.txt`。
- 打包：执行上述 PyInstaller 命令，产物保存于 `dist/LM-Translate-GUI/LM-Translate-GUI.exe`。
- 清理：删除 `build/` 和旧 EXE。
- 配置：确保 `dist/LM-Translate-GUI/gui_prefs.json` 存在并包含本次配置项（`theme: minty`；`pdf: { mode, vlm_url, vlm_key, dpi, pages }`）。EXE 启动读取该配置。

## 验证步骤
1. 启动 EXE：不再出现 `openai` 提示；默认主题 `minty` 已应用。
2. 主题切换：在下拉中选择其它主题，立即生效（无错误）。
3. PDF 面板：字段对齐一致，填写或留空 VLM 配置均正常；DPI 设置为 200/300 分别验证识别速度与质量差异。
4. 输出：正常写入目标文件（`.md/.txt`），翻译进度日志正常显示。