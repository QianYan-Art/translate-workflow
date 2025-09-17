# LM Studio 翻译工具

一个基于 LM Studio 的智能翻译工具，支持大文本分块翻译，提供图形界面和命令行两种使用方式。

## 功能特点

- 🤖 **AI 驱动翻译**：基于 LM Studio 本地大语言模型进行翻译
- 📝 **智能分块处理**：自动将大文本分割成合适的块进行翻译，避免超时
- 🔄 **上下文保持**：通过重叠字符保持翻译的连贯性
- 🎯 **断点续传**：支持从中断处继续翻译
- 🖥️ **双重界面**：提供友好的 GUI 界面和灵活的命令行工具
- 🔍 **分块预览**：可以预览和导出文本分块结果
- ⚡ **自动重试**：失败时自动重试，支持智能二分处理
- 🌐 **多语言支持**：支持多种语言间的翻译



## 安装和环境配置

### 前置要求

1. **Python 3.8+**
2. **LM Studio**：需要先安装并运行 LM Studio，加载合适的翻译模型
3. **虚拟环境**（推荐）

### 使用 uv 安装（推荐）

如果你已经安装了 [uv](https://github.com/astral-sh/uv)：

```bash
# 克隆项目
git clone <repository-url>
cd translate

# 创建虚拟环境并安装依赖
uv venv
uv pip install openai

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### 使用 pip 安装

```bash
# 克隆项目
git clone <repository-url>
cd translate

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 1. 启动 LM Studio

1. 打开 LM Studio
2. 加载一个支持翻译的模型（推荐多语言模型）
3. 启动本地服务器（默认端口 1234 或 1235）

### 2. 图形界面使用

```bash
python gui.py
```

图形界面功能：
- **翻译标签页**：配置翻译参数，选择输入输出文件
- **分块预览标签页**：预览文本如何被分块处理
- 支持实时日志显示和进度条

### 3. 命令行使用

#### 基本翻译

```bash
python translate_lmstudio.py
```

#### 自定义参数

```bash
python translate_lmstudio.py \
    --input "input.txt" \
    --output "output.txt" \
    --source-lang "en" \
    --target-lang "zh" \
    --chunk-size 2000 \
    --base-url "http://localhost:1234/v1"
```

#### 分块预览

```bash
python inspect_chunks.py --input "input.txt" --chunk-size 3000
```

## 配置说明

主要配置项在 `translate_lmstudio.py` 的 `CONFIG` 字典中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BASE_URL` | `http://localhost:1235/v1` | LM Studio API 地址 |
| `API_KEY` | `lm-studio` | API 密钥 |
| `MODEL` | `huihui-hunyuan-mt-chimera-7b-abliterated-i1` | 模型名称 |
| `CHUNK_SIZE_CHARS` | `3000` | 每块最大字符数 |
| `OVERLAP_CHARS` | `120` | 重叠字符数 |
| `SOURCE_LANG` | `auto` | 源语言（auto 为自动检测） |
| `TARGET_LANG` | `zh` | 目标语言 |
| `REQUEST_TIMEOUT` | `90` | 请求超时时间（秒） |
| `MAX_RETRIES` | `3` | 最大重试次数 |
| `AUTO_BISECT_ON_FAIL` | `True` | 失败时自动二分 |

## 高级功能

### 断点续传

如果翻译过程中断，可以设置 `RESUME=True` 继续翻译：

```bash
python translate_lmstudio.py --resume
```

### 自动二分处理

当某个文本块翻译失败时，工具会自动将其分成更小的块重试，直到成功或达到最小长度限制。

### 批量处理

可以通过脚本批量处理多个文件：

```python
import translate_lmstudio as tls

files = ["file1.txt", "file2.txt", "file3.txt"]
for file in files:
    tls.CONFIG["INPUT"] = file
    tls.CONFIG["OUTPUT"] = f"translated_{file}"
    # 调用翻译函数
```

## 故障排除

### 常见问题

1. **连接失败**：检查 LM Studio 是否正在运行，端口是否正确
2. **翻译质量差**：尝试调整 `CHUNK_SIZE_CHARS` 和 `OVERLAP_CHARS`
3. **超时错误**：增加 `REQUEST_TIMEOUT` 值
4. **内存不足**：减小 `CHUNK_SIZE_CHARS` 值

### 日志查看

程序运行时会显示详细的进度信息，包括：
- 当前处理的块编号
- 翻译进度百分比
- 错误信息和重试状态

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目采用 MIT 许可证。

## 更新日志

- **v1.0.0**：初始版本，支持基本翻译功能
- 添加了图形界面支持
- 实现了智能分块和断点续传
- 增加了自动二分处理功能