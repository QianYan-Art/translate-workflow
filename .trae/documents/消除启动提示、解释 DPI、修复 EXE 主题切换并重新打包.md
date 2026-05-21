## 问题与目标
- 消除 EXE 启动时的 “The 'openai' package is required...” 提示。
- 说明 PDF 识别选项中的 `DPI` 含义与建议取值。
- 修复打包后的 EXE 主题切换无效，并确保默认主题 `minty` 启动即生效。

## 解决方案
### 消除提示
- 调整 `translate_lmstudio.py` 的 `openai` 导入策略：改为在真正发起翻译时懒加载；在被 GUI 导入时不打印或退出。
- 具体：将顶层 `try: from openai import OpenAI ... except ImportError: print(...); sys.exit(1)` 改为：
  - 在 `run_translation()` 内部执行 `from openai import OpenAI`，若失败则仅在 CLI 启动或开始翻译时提示；GUI 导入不产生输出。

### DPI 解释
- `DPI`（dots per inch）用于将 PDF 页面渲染为位图供识别使用。
- 在实现中通过 `fitz.Matrix(dpi/72, dpi/72)` 进行缩放；默认 `200`。建议范围：
  - 150–200：速度与清晰度平衡；多数文档足够。
  - 300：更清晰但更慢，更占内存，适合小字体/扫描件。
- 取值越大图像越清晰、识别更稳，但处理更慢、内存更高。

### 修复 EXE 主题切换
- 代码层面：
  - 在 App 初始化时优先创建 `ttkbootstrap.Style`，使用 `PREFS['theme']`（缺省 `minty`）立即应用。
  - 主题下拉构建后主动调用一次应用方法，避免必须手动切换才生效。
- 打包层面：
  - 使用 PyInstaller 显式收集 `ttkbootstrap` 资源：`--collect-submodules ttkbootstrap --collect-data ttkbootstrap`（优于 `--collect-all`）。
  - 生成 EXE 后验证主题切换逻辑可用。

### 打包与配置
- 依赖管理：统一使用 `uv pip install -r requirements.txt`。
- 打包命令：`uvx pyinstaller --noconfirm --clean --onedir --collect-submodules ttkbootstrap --collect-data ttkbootstrap --name LM-Translate-GUI gui.py`。
- 清理中间文件：删除 `build/`。
- 配置加载：EXE 启动读取 `dist/LM-Translate-GUI/gui_prefs.json`；保留并更新 `pdf` 配置（`mode/vlm_url/vlm_key/dpi/pages`）与 `theme`（默认 `minty`）。

## 验证步骤
1. 启动 EXE：无 `openai` 提示；默认主题 `minty` 已应用。
2. 切换主题：从下拉选择其它主题，立即生效。
3. PDF 面板：字段对齐一致；识别模式与 VLM URL/Key、DPI、页范围工作正常。

## 执行计划
- 修改 `translate_lmstudio.py` 懒加载 `openai`，去除顶层退出提示。
- 调整 `gui.py` 主题初始化与应用调用；优化 PDF 面板网格对齐。
- 使用上述打包命令重新生成 EXE，删除中间文件，确保最终产物在 `dist`。
- 更新/写入 `dist/LM-Translate-GUI/gui_prefs.json`，同步本次新增配置项。

请确认后我将按上述步骤实施并交付可运行的 EXE。