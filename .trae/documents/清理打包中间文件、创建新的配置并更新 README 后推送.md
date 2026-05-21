## 目标
- 删除本地打包中间文件目录 `build/`。
- 在 `dist/LM-Translate-GUI/gui_prefs.json` 创建新的配置文件，EXE 启动从该 JSON 中读取配置。
- 更新 README 说明打包版配置读取与 DPI/VLM 配置使用方法，保持无表情、简洁样式。
- 提交并推送新的 README 与 `dist` 目录到 GitHub。

## 实施步骤
1) 删除打包中间文件：移除 `d:\MCP_Server\translate\build`。
2) 创建配置文件：新增 `dist/LM-Translate-GUI/gui_prefs.json`，包含
- `theme: "minty"`
- `defaults`: 基础参数（base_url、api_key、chunk 配置等）
- `pdf`: `{ mode, vlm_url, vlm_key, dpi, pages }`
3) 验证读取逻辑：打包模式下 `gui.py` 已优先从 EXE 同目录读取 `gui_prefs.json`，无需改动。
4) 更新 README：
- 在“打包程序”和“配置说明”中明确 EXE 从 `dist/LM-Translate-GUI/gui_prefs.json` 读取配置
- 解释 `DPI` 的作用与建议范围（150–200/300）
- 简述 VLM URL/Key 的填写与回退策略（auto/vlm/none）
5) Git 提交与推送：
- `git add README.md dist/LM-Translate-GUI/gui_prefs.json`
- `git commit -m "docs: update packaging config reading; add gui_prefs.json in dist"`
- `git push origin master`

## 交付
- 最新 README（无表情、含配置读取与 DPI/VLM 指南）
- `dist/LM-Translate-GUI/gui_prefs.json`（EXE 启动读取）
- 清理完成的工作区（无 build/ 中间文件）