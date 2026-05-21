## 目标
- 在现有翻译管线基础上，新增对 `.md`、`.pdf`、`.docx` 输入文件的支持。
- 保持分块、重叠上下文与错误重试逻辑不变，尽可能保留原始文档的段落/结构信息。

## 现状与切入点
- CLI 入口在 `translate_lmstudio.py:738-784`，读取输入文件位置在 `translate_lmstudio.py:508-512`（当前仅以纯文本读取）。
- GUI 仅允许选择 `.txt`，限制点在：
  - 文件选择与校验 `gui.py:585-589`、`gui.py:972-975`
  - 检查工具同样假设 `.txt`（可暂不扩展）。
- 分块/上下文逻辑已考虑 Markdown 代码围栏（`split_sentences` 保护 ```...```，见 `translate_lmstudio.py:51-63`）。

## 实现方案
### 文档解析
- 新增模块 `document_loader.py`，提供：
  - `load_document(path) -> str`：按扩展名路由，统一返回 UTF‑8 文本，段落以换行分隔。
  - `.txt/.md`：直接读取文本；`.md` 原样保留 Markdown 标记。
  - `.pdf`：优先使用 `pdfminer.six` 提取文本（按页拼接，保持段落换行）；必要时回退到 `PyPDF2` 的基本提取。
  - `.docx`：使用 `python-docx` 逐段落提取，保留空行分隔以反映结构。

### 管线集成
- 在 `translate_lmstudio.py:508-512` 用 `load_document(input_path)` 替换直接 `open()`：
  - 其余分块、重叠与上下文处理保持不变。
  - 对 `.md`，现有系统提示已要求保留 Markdown 格式（`translate_lmstudio.py:124-135`、`138-156`），无需额外处理。

### GUI 更新
- 输入文件选择器改为支持多格式：
  - `filetypes=[("Markdown", "*.md"), ("PDF", "*.pdf"), ("Word", "*.docx"), ("Text", "*.txt"), ("All", "*.*")]`
  - 移除 `.txt` 强制限制：修改 `gui.py:585-589` 与 `gui.py:972-975` 的校验逻辑，根据扩展名放行上述四类。
- 输出路径默认：
  - 继续允许用户指定任意目标文件；若输入为 `.md`，建议默认 `.md`；其他格式默认 `.txt`（简化首版交付）。

### 输出与格式保留策略
- 首版交付：统一输出为纯文本/Markdown：
  - `.md` 输入：建议输出 `.md`，保留标题/列表/代码块等结构（依赖模型遵守提示）。
  - `.pdf/.docx` 输入：输出 `.txt`，保持段落换行，便于审阅与后续再导入。
- 后续可选增强（第二阶段）：
  - 提供 `--export-docx` 将翻译结果写入新的 `.docx`（按段落拆分添加）。
  - PDF 原样版式重建复杂，建议仅提供简版文本 PDF 导出（可用 `reportlab`/`fpdf`）。

### 依赖管理
- 新增依赖：`pdfminer.six`、`python-docx`（以及备选 `PyPDF2`）。
- 在 `requirements.txt` 增补；GUI 打包配置沿用现有 PyInstaller 方案，添加对应 hook（环境中已有 `hook-pdfminer.py` 提示兼容性较好）。

### 测试与验证
- 构造三类样例：`sample.md`（含代码围栏/列表）、`sample.pdf`（段落+分页）、`sample.docx`（标题+正文）。
- 验证要点：
  - 分块统计与 GUI 进度正常更新（`Translating chunk X/Y`）。
  - `.md` 代码块与结构保留（对照原文）；`.pdf/.docx` 段落边界完整（换行位置合理）。
  - 失败自动二分与标记替换逻辑对多格式同样适用（`translate_lmstudio.py:557-595`、`358-403`）。

### 风险与边界
- PDF 文本提取质量受文档内容与字体嵌入影响，个别文档可能仅得无断行文本；通过 `LAParams` 调优并页级拼接降低风险。
- `.doc`（旧版 Word）不在首版支持范围；可提示先转 `.docx`。

## 交付物
- `document_loader.py` 新增。
- `translate_lmstudio.py`：替换输入读取为加载器调用，其他逻辑不变。
- `gui.py`：输入文件类型选择与校验更新；默认输出扩展智能化。
- `requirements.txt`：增补依赖。
- 基础样例与使用说明补充（README 小幅更新，仅涉及新格式支持说明）。