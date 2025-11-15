# LM Studio 翻译助手

<div align="center">

![Version](https://img.shields.io/badge/version-1.2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)

**基于 LM Studio 的智能翻译工具，提供现代化图形界面**

[功能特点](#-功能特点) • [快速开始](#-快速开始) • [使用指南](#-使用指南) • [配置说明](#-配置说明) • [打包程序](#-打包程序) • [更新日志](#-更新日志)

</div>

---

## 目录

- [功能特点](#-功能特点)
- [系统要求](#-系统要求)
- [安装配置](#-安装配置)
- [快速开始](#-快速开始)
- [使用指南](#-使用指南)
  - [图形界面使用](#图形界面使用)
  - [分块检查工具](#分块检查工具)
- [打包程序](#-打包程序)
- [配置说明](#-配置说明)
- [高级功能](#-高级功能)
  - [系统提示词自定义](#系统提示词自定义)
  - [LLM 超参数调节](#llm-超参数调节)
  - [自动二分处理](#自动二分处理)
  - [输出后处理](#输出后处理)
- [故障排除](#-故障排除)
- [更新日志](#-更新日志)
- [许可证](#-许可证)

---

## 功能特点

### 图形界面
- **图形界面**：直观的 GUI 界面，满足用户翻译需求
- **主题定制**：11 种精美主题可选（cosmo、litera、darkly 等）
- **智能交互**：实时进度显示、悬停效果、语义化按钮设计
- **目录记忆**：自动记住最近使用的输入/输出目录，减少重复选择

### 高级定制
- **提示词自定义**：支持自定义系统提示词和从文件加载
- **超参数调节**：完整的 LLM 参数控制（temperature、top_p、top_k 等）
- **分块策略**：智能文本分块，保持上下文连贯性

### 智能处理
- **自动二分**：失败时自动拆分长文本重试，提高成功率
- **断点续传**：支持从指定分块开始，避免重复翻译
- **流式响应**：减少长响应超时风险，提升稳定性

### 可靠性保障
- **错误处理**：多重重试机制和智能错误恢复
- **进度追踪**：实时显示翻译进度和状态信息
- **日志记录**：详细的操作日志便于问题诊断

---

## 系统要求

- **Python**: 3.9 或更高版本
- **操作系统**: Windows、Linux、macOS
- **LM Studio**: 已安装并运行的 LM Studio 服务
- **内存**: 建议 4GB 以上（取决于模型大小）

---

## 安装配置

### 1. 克隆项目
```bash
git clone https://github.com/your-username/lm-studio-translator.git
cd lm-studio-translator
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置 LM Studio
1. 启动 LM Studio
2. 加载您选择的翻译模型
3. 启动本地服务器（默认端口 1234 或 1235）
4. 记录服务器地址和模型名称

---

## 快速开始

### 图形界面
```bash
python gui.py
```

### 分块检查
```bash
python inspect_chunks.py --input "input.txt" --chunks "1-5"
```

---

## 使用指南

### 图形界面使用

#### 主题选择
- 在界面顶部选择您喜欢的主题
- 支持 11 种主题：cosmo（蓝白简洁）、litera（文档风）、darkly（深色）等
- 主题选择后立即生效

#### 文件配置
1. **输入文件**：选择待翻译的文本文件
2. **输出文件**：指定翻译结果保存路径
3. **语言设置**：配置源语言（auto 自动检测）和目标语言
4. **目录记忆**：自动记录并默认回填最近使用的输入/输出目录

#### 参数调整
- **分块大小**：控制每次翻译的文本长度（默认 3000 字符）
- **重叠字符**：设置分块间的重叠部分（默认 120 字符）
- **起始分块**：从指定分块开始翻译（用于断点续传）

#### 高级选项
- **提示词自定义**：在多行文本框中输入自定义系统提示词
- **超参数调节**：调整 temperature、top_p、top_k 等 LLM 参数
- **处理选项**：配置重试次数、超时时间、错误处理策略

## 打包程序

### dist 文件夹说明

`dist` 文件夹包含已打包的可执行程序与配置文件，无需安装 Python 环境即可直接运行：

```
dist/
└── LM-Translate-GUI/
    ├── _internal/              # 程序依赖文件夹（运行时与库）
    ├── LM-Translate-GUI.exe    # 主程序可执行文件
    └── gui_prefs.json          # 程序启动配置（主题/识别/参数）
```

### 使用方法

1. **直接运行**：双击 `LM-Translate-GUI.exe` 启动图形界面（启动时读取同目录 `gui_prefs.json`）
2. **便携使用**：整个 `LM-Translate-GUI` 文件夹可以复制到任何位置使用
3. **系统要求**：Windows 10/11 x64 系统

### 配置与注意事项

- 首次运行可能需要几秒钟加载时间
- 确保 LM Studio 服务正在运行
- 程序会自动保存界面设置到 `gui_prefs.json`
- 如需修改默认主题与识别配置，直接编辑 `dist/LM-Translate-GUI/gui_prefs.json`
- 打包模式下不依赖全局 Python 与包管理

### 分块检查工具

检查文本如何被分块，优化翻译效果：

```bash
# 检查特定分块
python inspect_chunks.py --input "input.txt" --chunks "1,5,10-15"

# 调整分块参数
python inspect_chunks.py --input "input.txt" --chunk-size 2000 --overlap 150
```

---

## 配置说明

### 环境变量配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `BASE_URL` | `http://localhost:1235/v1` | LM Studio 服务地址 |
| `API_KEY` | `lm-studio` | API 密钥（LM Studio 接受任意值） |
| `MODEL` | `huihui-hunyuan-mt-chimera-7b-abliterated-i1` | 模型名称 |
| `SOURCE_LANG` | `auto` | 源语言（auto 自动检测） |
| `TARGET_LANG` | `zh` | 目标语言 |
| `CHUNK_SIZE_CHARS` | `3000` | 分块最大字符数 |
| `OVERLAP_CHARS` | `120` | 分块重叠字符数 |
| `REQUEST_TIMEOUT` | `90` | 请求超时时间（秒） |
| `MAX_RETRIES` | `3` | 最大重试次数 |
| `TEMPERATURE` | `0.0` | 采样温度（0.0-2.0） |
| `TOP_P` | `1.0` | 核采样参数（0-1） |
| `TOP_K` | `0` | Top-K 采样（0 表示禁用） |
| `REPETITION_PENALTY` | `0.0` | 重复惩罚（0 表示禁用） |
| `LENGTH_PENALTY` | `0.0` | 长度惩罚（0 表示禁用） |
| `CHAIN_TAG` | `think` | 思维链标签名（如 `think`=>`<think></think>`） |

### 布尔选项配置

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `OVERWRITE` | `False` | 是否覆盖已存在的输出文件 |
| `RESUME` | `True` | 是否支持断点续传 |
| `CONTEXT_ONLY_OVERLAP` | `True` | 重叠部分仅作上下文（不翻译） |
| `SKIP_ON_ERROR` | `True` | 出错后自动跳过并标记 |
| `USE_STREAM` | `True` | 使用流式响应 |
| `AUTO_BISECT_ON_FAIL` | `True` | 失败时自动二分重试 |
| `HIDE_CHAIN` | `False` | 隐藏思维链块（如 `<think>...</think>`） |

---

## 🔬 高级功能

### 系统提示词自定义

在"提示词与超参数"区域的多行文本框中输入自定义提示词。

### LLM 超参数调节

#### 参数说明
- **temperature** (0.0-2.0)：控制输出随机性，0 为确定性输出，值越高越随机
- **top_p** (0-1)：核采样，控制候选词汇范围，建议 0.8-0.95
- **top_k** (整数)：限制每步考虑的词汇数量，0 表示不限制
- **repetition_penalty** (浮点数)：重复惩罚，减少重复内容
- **length_penalty** (浮点数)：长度惩罚，影响输出长度偏好

#### 使用建议
- **翻译任务**：temperature=0.1-0.3, top_p=0.9
- **创意翻译**：temperature=0.5-0.7, top_p=0.8
- **技术文档**：temperature=0.0-0.1, top_p=0.95

### 自动二分处理

当翻译失败时，系统会自动将长文本按语义边界拆分为更小的片段重试：

- **BISECT_MIN_CHARS** (600)：最小分块长度，低于此值不再拆分
- **BISECT_MAX_DEPTH** (3)：最大递归深度，防止无限拆分
- **语义边界**：优先在句号、换行符等自然断点处拆分

### 输出后处理

- 隐藏思维链：启用 `--hide-chain` 后移除如 `<think>...</think>` 的内容，可用 `--chain-tag` 指定自定义标签（例如 `thinking`）。
- 背景提示清理：自动去除“（以下内容仅做背景信息，不输出）…（以下内容是正文）”提示段，以及与重叠上下文相关的标识（如“仅将重叠部分作为上下文，不输出”）的残留文本；同时支持英文从 `Context (for reference only...)` 到 `Content to translate (OUTPUT ONLY...)` 的提示段清理。
- GUI 支持：在“开关选项”勾选“隐藏思维链”，并在“输出后处理”区域设置标签名。

```bash
python translate_lmstudio.py --hide-chain --chain-tag think \
  --input whole.txt --output translated.txt
```

---

## 故障排除

### 常见问题

#### 1. 连接 LM Studio 失败
```
错误：Connection refused
解决：检查 LM Studio 是否启动，端口是否正确
```

#### 2. 模型名称错误
```
错误：Model not found
解决：在 LM Studio 中确认模型名称，更新配置
```

#### 3. 翻译超时
```
错误：Request timeout
解决：增加 REQUEST_TIMEOUT 值或减少 CHUNK_SIZE_CHARS
```

#### 4. 内存不足
```
错误：Out of memory
解决：减少分块大小，关闭其他应用程序
```

### 性能优化

1. **调整分块大小**：根据模型性能和内存情况调整
2. **启用流式响应**：减少超时风险
3. **合理设置重试**：平衡成功率和速度
4. **使用 GPU 加速**：在 LM Studio 中启用 GPU

---

## 更新日志

### v1.2.0 (2025-10-22) - 上下文标识清理与目录记忆
- 清理重叠上下文标识残留（“仅将重叠部分作为上下文，不输出”）
- 目录记忆：记录并默认回填最近使用的输入/输出目录
- 思维链隐藏：支持 `--hide-chain` 与 `--chain-tag`（CLI/GUI）
- 文档更新：README 区分 1.2.0 与 1.1.1，补充用法说明
- 最低 Python 版本更新为 3.9+（兼容 BooleanOptionalAction）

### v1.1.1 (2025-09-21) - 输出后处理与打包清理
- 界面升级：集成 ttkbootstrap，提供 11 种主题
- 功能增强：新增系统提示词自定义和 LLM 超参数调节
- 交互优化：添加悬停效果、焦点反馈和语义化按钮
- 布局改进：优化控件布局和间距
- 依赖更新：添加 ttkbootstrap 依赖

### v1.0.0 (2025-09-17) - 初始版本
- 核心功能：基于 LM Studio 的文本翻译
- 图形界面：现代化的 GUI 界面
- 智能分块：自动文本分块和上下文处理
- 错误处理：重试机制和自动二分功能
- 分块检查：独立的分块预览工具

---

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

---

<div align="center">



如果这个项目对您有帮助，请给个 Star

[报告问题](https://github.com/your-username/lm-studio-translator/issues) • [功能建议](https://github.com/your-username/lm-studio-translator/issues) • [贡献代码](https://github.com/your-username/lm-studio-translator/pulls)

</div>
### 打包版配置文件（gui_prefs.json）

位于 `dist/LM-Translate-GUI/gui_prefs.json`，程序启动时读取。

- `theme`: 启动主题，推荐 `minty`
- `defaults`: 基本参数（如 base_url、api_key、chunk/overlap 等），为空时使用内置默认
- `pdf`: 识别配置
  - `mode`: `auto`（优先用 VLM）/`vlm`/`none`
  - `vlm_url`: 远程 VLM 服务地址
  - `vlm_key`: 远程 VLM 服务密钥
  - `dpi`: PDF 渲染分辨率，默认 200（150–200 平衡、300 更清晰但更慢）
  - `pages`: 页范围，例如 `1-3,7`