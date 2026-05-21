## 目标
- 当选择 PDF 翻译时，先进行一次“识别”（OCR 或 VLM），将页面内容转换为纯文本或 Markdown，再交给现有 LLM 翻译管线。
- 在 GUI 中，当选择到 PDF 文档时自动显示填写识别服务 URL 与 Key 的输入框；未填写则走默认（当前纯文本提取）。

## 识别方式与优先级
- 识别方式：`ocr`、`vlm`、`auto`、`none`（默认 `auto`）。
- 优先级策略（`auto`）：
  - 若提供了 `VLM_URL/KEY` → 用 VLM 识别。
  - 否则若提供了 `OCR_URL/KEY` → 用 OCR 识别。
  - 否则走默认纯文本提取（现有 pdfminer/PyPDF2）。

## 技术方案
### 文档渲染（转图）
- 将 PDF 页渲染为图像（PNG/JPEG），供 OCR/VLM 输入。
- 首选 `PyMuPDF`（可靠、无需外部 poppler），或退回 `pdf2image`（需 Poppler）。
- 设置可调 `dpi`（默认 200）与页范围（如 `1-5,7`）。

### OCR（远程优先，本地可选）
- 远程 OCR：向 `OCR_URL` 发送每页图像（多部分或 base64），携带 `Authorization: Bearer <OCR_KEY>`，返回文本。
- 本地 OCR（可选）：`pytesseract`（需安装 Tesseract），仅作为后备；默认不启用。

### VLM 调用
- VLM 接口：向 `VLM_URL` 发送每页图像与提示语（"读取页面中的文字，输出纯文本"），携带 `Authorization: Bearer <VLM_KEY>`。
- 支持 OpenAI 兼容式多模态或自定义 REST；以 JSON 请求包含 base64 图像数组与指令文本。

### 文本合并与清洗
- 页级结果按顺序拼接，页间插入分页分隔（如 `\n\n--- Page N ---\n\n`）。
- 规范换行与空白；若选择 Markdown 输出，保留基本结构（标题、列表）由识别结果决定。

## 管线集成点
- 文档加载器：扩展 `document_loader.load_document(path, *, pdf_recognition)`：
  - `pdf_recognition` 包含：`mode`（ocr/vlm/auto/none）、`ocr_url`、`ocr_key`、`vlm_url`、`vlm_key`、`dpi`、`pages`。
  - `.pdf` 时根据配置走识别流程，否则回退到现有提取。
- 翻译入口：在 `translate_lmstudio.py` 增加 CLI/CONFIG：
  - `--pdf-recognizer {auto,ocr,vlm,none}`、`--ocr-url`、`--ocr-key`、`--vlm-url`、`--vlm-key`、`--pdf-dpi`、`--pdf-pages`。
  - 调用 `load_document(input_path, pdf_recognition=ns...)`（位置：读取输入的集成点在 translate_lmstudio.py:508-512）。
- GUI：当选中文件扩展为 `.pdf` 时自动显示“PDF 识别选项”面板：
  - 字段：识别方式（下拉/单选：自动/OCR/VLM/禁用）、`URL`、`API Key`、可选 `DPI`、`页范围`。
  - 未填写 URL/Key 时沿用默认（纯文本提取）。
  - 启用与隐藏逻辑挂在文件选择回调（相关位置：`gui.py:585-605` 文件选择器；启动前校验 `gui.py:972-975`）。

## 默认行为与回退
- 未提供 URL/Key：保持当前默认（pdfminer/PyPDF2 文本提取）。
- 页渲染失败或识别异常：写入日志并回退到默认文本提取，保障可用性。

## 安全与性能
- 不在日志打印密钥；GUI 不回显 Key 内容（mask）。
- 批量分页调用时限速与重试，避免服务过载；可控超时配置。
- 控制 DPI 与页范围，减少开销；大文档分页进度提示复用现有日志行（"Translating chunk X/Y" 前加入识别进度）。

## 依赖与打包
- 新增依赖：`pymupdf`（或选择 `pdf2image` + `pytesseract` 作为备选）。
- 保持远程 OCR/VLM 默认路径，无需本地引擎时不新增体积。
- 打包时添加相应 hook；密钥不写入配置文件。

## 测试与验证
- 测试样例：扫描型 PDF（需 OCR/VLM）、文本型 PDF（走默认提取）。
- 用 GUI 选择 PDF，分别填写 OCR 与 VLM 的 URL/Key 与留空场景，确认：
  - 识别面板自动显示与隐藏。
  - 未填时回退到默认；填了能调用识别并输出文本。
  - 翻译进度与分块逻辑保持正常。

## 交付项
- 新增 `pdf_recognizer.py`（渲染与远程识别调用封装）。
- 扩展 `document_loader.py` 接口与 PDF 识别路径。
- 更新 `translate_lmstudio.py` 的 CLI/CONFIG 与识别集成。
- 更新 `gui.py` 增加 PDF 识别选项面板与交互逻辑。
- 根据最终选择补充 `requirements.txt` 依赖。
