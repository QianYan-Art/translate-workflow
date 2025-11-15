# -*- coding: utf-8 -*-
# Simple GUI wrapper for translate_lmstudio.py and inspect_chunks.py
# Designed for non-technical users on Windows. No extra dependencies required.

import os
import sys
import threading
import queue
import subprocess
import io
import contextlib
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json

# Use executable directory when packaged (PyInstaller), else file directory
if getattr(sys, 'frozen', False):
    SCRIPT_DIR = Path(sys.executable).resolve().parent
else:
    SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PY = (SCRIPT_DIR / ".venv" / "Scripts" / "python.exe")
PYTHON_EXE = str(DEFAULT_PY if DEFAULT_PY.exists() else sys.executable)

# Preferences: remember last used directory
PREFS_FILE = SCRIPT_DIR / "gui_prefs.json"

def _load_prefs() -> dict:
    """
    加载首选项：优先加载打包目录 dist/LM-Translate-GUI/gui_prefs.json，
    再叠加当前脚本目录下的 gui_prefs.json（如存在），后者覆盖前者。
    可在 prefs["paths"] 中提供 "input_dir" 与 "output_dir" 来分别记忆两个浏览路径。
    """
    merged: dict = {}
    try:
        dist_prefs = (
            (SCRIPT_DIR / "gui_prefs.json") if getattr(sys, 'frozen', False)
            else (SCRIPT_DIR / "dist" / "LM-Translate-GUI" / "gui_prefs.json")
        )
        if dist_prefs.exists():
            with open(dist_prefs, "r", encoding="utf-8") as f:
                d = json.load(f) or {}
                if isinstance(d, dict):
                    merged.update(d)
    except Exception:
        pass
    try:
        if PREFS_FILE.exists():
            with open(PREFS_FILE, "r", encoding="utf-8") as f:
                d = json.load(f) or {}
                if isinstance(d, dict):
                    merged.update(d)
    except Exception:
        pass
    return merged

def _save_prefs(d: dict) -> None:
    try:
        with open(PREFS_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

PREFS = _load_prefs()

# Best-effort: try to read sane defaults; avoid hard import failures
DEFAULTS = {
    "base_url": "http://localhost:1235/v1",
    "api_key": "lm-studio",
    "model": "",
    "input": "whole.txt",
    "output": "translated.txt",
    "source_lang": "auto",
    "target_lang": "zh",
    "chunk_size": 3000,
    "overlap_chars": 120,
    "context_only_overlap": True,
    "start_chunk": 1,
    "request_timeout": 90,
    "max_retries": 3,
    "skip_on_error": True,
    "use_stream": True,
    "auto_bisect": True,
    "bisect_min_chars": 600,
    "bisect_max_depth": 3,
    # 输出后处理默认值
    "hide_chain": False,
    "chain_tag": "think",
}

try:
    # Prefer a normal import so PyInstaller can analyze and include the module
    import translate_lmstudio as tlm_for_defaults
    cfg = getattr(tlm_for_defaults, "CONFIG", {})
    DEFAULTS.update({
        "base_url": cfg.get("BASE_URL", DEFAULTS["base_url"]),
        "api_key": cfg.get("API_KEY", DEFAULTS["api_key"]),
        "model": cfg.get("MODEL", DEFAULTS["model"]),
        "input": cfg.get("INPUT", DEFAULTS["input"]),
        "output": cfg.get("OUTPUT", DEFAULTS["output"]),
        "source_lang": cfg.get("SOURCE_LANG", DEFAULTS["source_lang"]),
        "target_lang": cfg.get("TARGET_LANG", DEFAULTS["target_lang"]),
        "chunk_size": cfg.get("CHUNK_SIZE_CHARS", DEFAULTS["chunk_size"]),
        "overlap_chars": cfg.get("OVERLAP_CHARS", DEFAULTS["overlap_chars"]),
        "context_only_overlap": cfg.get("CONTEXT_ONLY_OVERLAP", DEFAULTS["context_only_overlap"]),
        "start_chunk": cfg.get("START_CHUNK", DEFAULTS["start_chunk"]),
        "request_timeout": cfg.get("REQUEST_TIMEOUT", DEFAULTS["request_timeout"]),
        "max_retries": cfg.get("MAX_RETRIES", DEFAULTS["max_retries"]),
        "skip_on_error": cfg.get("SKIP_ON_ERROR", DEFAULTS["skip_on_error"]),
        "use_stream": cfg.get("USE_STREAM", DEFAULTS["use_stream"]),
        "auto_bisect": cfg.get("AUTO_BISECT_ON_FAIL", DEFAULTS["auto_bisect"]),
        "bisect_min_chars": cfg.get("BISECT_MIN_CHARS", DEFAULTS["bisect_min_chars"]),
        "bisect_max_depth": cfg.get("BISECT_MAX_DEPTH", DEFAULTS["bisect_max_depth"]),
        # 输出后处理
        "hide_chain": cfg.get("HIDE_CHAIN", DEFAULTS["hide_chain"]),
        "chain_tag": cfg.get("CHAIN_TAG", DEFAULTS["chain_tag"]),
    })
except SystemExit:
    # openai missing during import; fall back to built-in defaults
    pass
except Exception:
    pass

# Update DEFAULTS with values from gui_prefs.json if available
prefs_defaults = PREFS.get("defaults", {})
if prefs_defaults:
    DEFAULTS.update(prefs_defaults)

class ProcessRunner:
    def __init__(self, text_widget: tk.Text, progressbar: ttk.Progressbar, status_var: tk.StringVar):
        self.text = text_widget
        self.pb = progressbar
        self.status = status_var
        self.proc = None
        self.q = queue.Queue()
        self.reader_threads = []
        self.stop_requested = False
        self.total_chunks = None
        self.current_chunk = 0

    def _append_text(self, s: str):
        self.text.configure(state=tk.NORMAL)
        self.text.insert(tk.END, s)
        self.text.see(tk.END)
        self.text.configure(state=tk.DISABLED)

    def _enqueue_reader(self, stream, prefix=""):
        def reader():
            for line in iter(stream.readline, ""):
                if not line:
                    break
                self.q.put(prefix + line)
            stream.close()
        t = threading.Thread(target=reader, daemon=True)
        t.start()
        self.reader_threads.append(t)

    def _pulser(self):
        try:
            while True:
                line = self.q.get_nowait()
                # very lightweight progress parsing
                if "Translating chunk" in line:
                    # format: Translating chunk X/Y (content_chars=..., ctx_chars=...)
                    try:
                        part = line.split("Translating chunk", 1)[1].strip()
                        num, rest = part.split(" ", 1)
                        cur, total = num.split("/")
                        self.current_chunk = int(cur)
                        self.total_chunks = int(total)
                        if self.total_chunks > 0:
                            self.pb.config(maximum=self.total_chunks)
                            self.pb['value'] = self.current_chunk
                            self.status.set(f"进度：{self.current_chunk}/{self.total_chunks}")
                    except Exception:
                        pass
                self._append_text(line)
        except queue.Empty:
            pass
        if self.proc and self.proc.poll() is None:
            self.text.after(80, self._pulser)
        else:
            # process ended
            self.pb['value'] = self.current_chunk if self.total_chunks else 0
            code = None if not self.proc else self.proc.returncode
            if code == 0:
                self.status.set("完成（退出码 0）")
            else:
                self.status.set(f"已结束（退出码 {code}）")

    def start(self, args_list):
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("提示", "已有任务在执行，请先停止或等待完成。")
            return
        self.text.configure(state=tk.NORMAL)
        self.text.delete(1.0, tk.END)
        self.text.configure(state=tk.DISABLED)
        self.pb['value'] = 0
        self.status.set("正在启动...")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            self.proc = subprocess.Popen(
                [PYTHON_EXE] + args_list,
                cwd=str(SCRIPT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
        except FileNotFoundError:
            messagebox.showerror("错误", f"找不到 Python 解释器：\n{PYTHON_EXE}")
            self.status.set("启动失败")
            return
        except Exception as e:
            messagebox.showerror("错误", f"启动失败：{e}")
            self.status.set("启动失败")
            return
        # start readers and pulser
        self.reader_threads = []
        self.q = queue.Queue()
        self._enqueue_reader(self.proc.stdout, "")
        self._enqueue_reader(self.proc.stderr, "[ERR] ")
        self.text.after(80, self._pulser)
        self.status.set("运行中...")

    def stop(self):
        if not self.proc or self.proc.poll() is not None:
            self.status.set("无运行中的任务")
            return
        self.status.set("正在停止...")
        try:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        except Exception:
            pass
        self.status.set("已停止")

# New: In-process runner for frozen (packaged) mode
class InProcessRunner:
    def __init__(self, text_widget: tk.Text, progressbar: ttk.Progressbar, status_var: tk.StringVar):
        self.text = text_widget
        self.pb = progressbar
        self.status = status_var
        self.q = queue.Queue()
        self.thread = None
        self.total_chunks = None
        self.current_chunk = 0

    def _append_text(self, s: str):
        self.text.configure(state=tk.NORMAL)
        self.text.insert(tk.END, s)
        self.text.see(tk.END)
        self.text.configure(state=tk.DISABLED)

    def _pulser(self):
        try:
            while True:
                line = self.q.get_nowait()
                if "Translating chunk" in line:
                    try:
                        part = line.split("Translating chunk", 1)[1].strip()
                        num, rest = part.split(" ", 1)
                        cur, total = num.split("/")
                        self.current_chunk = int(cur)
                        self.total_chunks = int(total)
                        if self.total_chunks > 0:
                            self.pb.config(maximum=self.total_chunks)
                            self.pb['value'] = self.current_chunk
                            self.status.set(f"进度：{self.current_chunk}/{self.total_chunks}")
                    except Exception:
                        pass
                self._append_text(line)
        except queue.Empty:
            pass
        if self.thread and self.thread.is_alive():
            self.text.after(80, self._pulser)
        else:
            self.pb['value'] = self.current_chunk if self.total_chunks else 0
            self.status.set("完成")

    def start_translate(self, kwargs: dict, overwrite: bool, resume: bool):
        if self.thread and self.thread.is_alive():
            messagebox.showinfo("提示", "已有任务在执行，请先等待完成。")
            return
        self.text.configure(state=tk.NORMAL)
        self.text.delete(1.0, tk.END)
        self.text.configure(state=tk.DISABLED)
        self.pb['value'] = 0
        self.status.set("正在启动...")

        runner = self
        class Writer:
            def __init__(self, q: queue.Queue):
                self.q = q
                self._buf = ""
            def write(self, s: str):
                if not isinstance(s, str):
                    s = str(s)
                self._buf += s
                while "\n" in self._buf:
                    line, self._buf = self._buf.split("\n", 1)
                    self.q.put(line + "\n")
            def flush(self):
                if self._buf:
                    self.q.put(self._buf)
                    self._buf = ""

        def target():
            w = Writer(runner.q)
            try:
                with contextlib.redirect_stdout(w), contextlib.redirect_stderr(w):
                    import translate_lmstudio as tlm
                    tlm.OVERWRITE_FLAG = bool(overwrite)
                    tlm.RESUME_FLAG = bool(resume)
                    tlm.run_translation(**kwargs)
            except BaseException as e:
                # Ensure error is reported (including SystemExit)
                try:
                    import traceback
                    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                except Exception:
                    tb = str(e)
                runner.q.put(f"[ERR] {tb}\n")

        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()
        self.text.after(80, self._pulser)
        self.status.set("运行中...")

    def stop(self):
        # Cooperative stop is not supported without modifying translator; inform user
        messagebox.showinfo("提示", "打包模式下暂不支持强制停止，请等待当前分块完成或关闭程序。")

# -------- New: Scrollable container for small screens --------
class ScrollableFrame(ttk.Frame):
    """A vertically scrollable frame using a Canvas + inner frame.
    Put content into self.content.
    """
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.inner = ttk.Frame(self.canvas)
        self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Resize scrollregion when inner changes size
        def _on_configure(event=None):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            # Keep inner width equal to canvas width
            try:
                self.canvas.itemconfigure(self.inner_id, width=self.canvas.winfo_width())
            except Exception:
                pass
        self.inner.bind("<Configure>", _on_configure)
        self.canvas.bind("<Configure>", _on_configure)

        # Mousewheel support (bind on enter, unbind on leave)
        def _on_mousewheel(event):
            try:
                # Linux: Button-4 (up) / Button-5 (down)
                if getattr(event, 'num', None) == 4:
                    self.canvas.yview_scroll(-1, "units")
                    return
                if getattr(event, 'num', None) == 5:
                    self.canvas.yview_scroll(1, "units")
                    return
                # Windows / macOS: <MouseWheel> with delta
                delta = int(getattr(event, 'delta', 0))
                if delta == 0:
                    return
                step = -1 if delta > 0 else 1  # Windows: +120 上滚，-120 下滚
                self.canvas.yview_scroll(step, "units")
            except Exception:
                pass
        def _bind_wheel(_=None):
            try:
                self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
                self.canvas.bind_all("<Button-4>", _on_mousewheel)
                self.canvas.bind_all("<Button-5>", _on_mousewheel)
            except Exception:
                pass
        def _unbind_wheel(_=None):
            try:
                self.canvas.unbind_all("<MouseWheel>")
                self.canvas.unbind_all("<Button-4>")
                self.canvas.unbind_all("<Button-5>")
            except Exception:
                pass
        self.canvas.bind("<Enter>", _bind_wheel)
        self.canvas.bind("<Leave>", _unbind_wheel)
        self.inner.bind("<Enter>", _bind_wheel)
        self.inner.bind("<Leave>", _unbind_wheel)

    @property
    def content(self):
        return self.inner

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LMStudio 翻译助手 GUI")
        self.geometry("980x680")
        self.minsize(900, 600)

        # 优化主题与基础样式：在 Windows 用 vista，其它平台退回 clam；统一一些控件的 padding
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except tk.TclError:
            try:
                style.theme_use("clam")
            except tk.TclError:
                pass
            style.configure("TNotebook.Tab", padding=(10, 4))
            style.configure("TButton", padding=(8, 4))
            style.configure("TCheckbutton", padding=(2, 2))
            style.configure("TLabelframe", padding=(8, 6))
            style.configure("TLabelframe.Label", padding=(4, 0))
            try:
                if getattr(self, "_tb_style", None):
                    self._apply_custom_styles(self._tb_style)
            except Exception:
                pass
        # 创建 Notebook（标签页容器）
        self.notebook = ttk.Notebook(self)
        self.tab_translate = ttk.Frame(self.notebook)
        self.tab_inspect = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_translate, text="开始翻译")
        self.notebook.add(self.tab_inspect, text="检查分块")
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # last used directory in memory
        self._last_dir = PREFS.get("last_dir") or str(SCRIPT_DIR)

        self._build_translate_tab()
        self._build_inspect_tab()

    def _get_initial_dir(self, kind: str | None = None) -> str:
        """根据类型返回默认目录：
        - kind == "input": 使用 PREFS.paths.input_dir
        - kind == "output": 使用 PREFS.paths.output_dir
        - 其他/缺省：fallback 到 last_dir 或脚本目录
        """
        try:
            paths = PREFS.get("paths", {}) if isinstance(PREFS, dict) else {}
            if kind == "input" and paths.get("input_dir"):
                return str(Path(paths["input_dir"]))
            if kind == "output" and paths.get("output_dir"):
                return str(Path(paths["output_dir"]))
        except Exception:
            pass
        d = self._last_dir or PREFS.get("last_dir") or str(SCRIPT_DIR)
        try:
            return str(Path(d))
        except Exception:
            return str(SCRIPT_DIR)

    def _remember_path(self, selected_path: str) -> None:
        try:
            if not selected_path:
                return
            p = Path(selected_path)
            d = p.parent if p.suffix else (p if p.is_dir() else p.parent)
            if not d:
                return
            self._last_dir = str(d)
            PREFS["last_dir"] = self._last_dir
            _save_prefs(PREFS)
        except Exception:
            pass

    def _remember_input_path(self, selected_file: str) -> None:
        try:
            if not selected_file:
                return
            p = Path(selected_file)
            d = p.parent if p.exists() or p.suffix else (p if p.is_dir() else p.parent)
            if not d:
                return
            PREFS.setdefault("paths", {})["input_dir"] = str(d)
            _save_prefs(PREFS)
        except Exception:
            pass

    def _remember_output_path(self, selected_file: str) -> None:
        try:
            if not selected_file:
                return
            p = Path(selected_file)
            d = p.parent if p.suffix else (p if p.is_dir() else p.parent)
            if not d:
                return
            PREFS.setdefault("paths", {})["output_dir"] = str(d)
            _save_prefs(PREFS)
        except Exception:
            pass

    def _build_translate_tab(self):
        # Use scrollable container to improve small-screen usability
        sc = ScrollableFrame(self.tab_translate)
        sc.pack(fill=tk.BOTH, expand=True)
        frm = sc.content
        pad = {"padx": 6, "pady": 4}

        # Variables
        self.var_base = tk.StringVar(value=DEFAULTS["base_url"])
        self.var_key = tk.StringVar(value=DEFAULTS["api_key"])
        self.var_model = tk.StringVar(value=DEFAULTS["model"])
        self.var_input = tk.StringVar(value=DEFAULTS["input"])
        self.var_output = tk.StringVar(value=DEFAULTS["output"])
        self.var_src = tk.StringVar(value=DEFAULTS["source_lang"])
        self.var_tgt = tk.StringVar(value=DEFAULTS["target_lang"])
        self.var_chunk = tk.IntVar(value=DEFAULTS["chunk_size"])
        self.var_overlap = tk.IntVar(value=DEFAULTS["overlap_chars"])
        self.var_ctx_only = tk.BooleanVar(value=DEFAULTS["context_only_overlap"])
        self.var_start = tk.IntVar(value=DEFAULTS["start_chunk"])
        self.var_timeout = tk.IntVar(value=DEFAULTS["request_timeout"])
        self.var_retries = tk.IntVar(value=DEFAULTS["max_retries"])
        self.var_skip = tk.BooleanVar(value=DEFAULTS["skip_on_error"])
        self.var_stream = tk.BooleanVar(value=DEFAULTS["use_stream"])
        self.var_bisect = tk.BooleanVar(value=DEFAULTS["auto_bisect"])
        self.var_minchars = tk.IntVar(value=DEFAULTS["bisect_min_chars"])
        self.var_maxdepth = tk.IntVar(value=DEFAULTS["bisect_max_depth"])
        self.var_overwrite = tk.BooleanVar(value=False)
        self.var_resume = tk.BooleanVar(value=True)
        # 新增：隐藏思维链与自定义标签
        self.var_hide_chain = tk.BooleanVar(value=DEFAULTS.get("hide_chain", False))
        self.var_chain_tag = tk.StringVar(value=DEFAULTS.get("chain_tag", "think"))

        # New: LLM system prompt & hyperparameters (empty means use defaults)
        self.var_temperature = tk.StringVar(value=DEFAULTS.get("temperature", ""))
        self.var_top_p = tk.StringVar(value=DEFAULTS.get("top_p", ""))
        self.var_top_k = tk.StringVar(value=DEFAULTS.get("top_k", ""))
        self.var_repetition_penalty = tk.StringVar(value=DEFAULTS.get("repetition_penalty", ""))
        self.var_length_penalty = tk.StringVar(value=DEFAULTS.get("length_penalty", ""))
        pdf_cfg = PREFS.get("pdf", {}) if isinstance(PREFS, dict) else {}
        self.var_pdf_mode = tk.StringVar(value=(pdf_cfg.get("mode") or "auto"))
        self.var_vlm_url = tk.StringVar(value=(pdf_cfg.get("vlm_url") or ""))
        self.var_vlm_key = tk.StringVar(value=(pdf_cfg.get("vlm_key") or ""))
        try:
            self.var_pdf_dpi = tk.IntVar(value=int(pdf_cfg.get("dpi") or 200))
        except Exception:
            self.var_pdf_dpi = tk.IntVar(value=200)
        self.var_pdf_pages = tk.StringVar(value=(pdf_cfg.get("pages") or ""))

        # Grid: 3 columns labels/entries/buttons
        row = 0
        
        # 主题选择
        theme_frame = ttk.LabelFrame(frm, text="界面主题", padding=(8, 6))
        theme_frame.grid(row=row, column=0, columnspan=4, sticky="ew", **pad)
        self.var_theme = tk.StringVar(value=PREFS.get("theme", "minty"))
        ttk.Label(theme_frame, text="选择主题").grid(row=0, column=0, sticky="w", padx=(0, 8))
        theme_options = ["cosmo", "litera", "lumen", "flatly", "united", "darkly", "cyborg", "superhero", "solar", "vapor", "minty"]
        self.theme_combo = ttk.Combobox(theme_frame, textvariable=self.var_theme, values=theme_options, state="readonly", width=15)
        self.theme_combo.grid(row=0, column=1, sticky="w", padx=(0, 8))
        self.theme_combo.bind("<<ComboboxSelected>>", self._on_theme_change)
        try:
            self._on_theme_change()
        except Exception:
            pass
        theme_frame.columnconfigure(2, weight=1)
        row += 1
        
        def add_entry(label, var, width=50, browse=None, col_span=2):
            nonlocal row
            ttk.Label(frm, text=label).grid(row=row, column=0, sticky="e", **pad)
            ent = ttk.Entry(frm, textvariable=var, width=width)
            ent.grid(row=row, column=1, sticky="we", **pad, columnspan=col_span)
            if browse:
                ttk.Button(frm, text="浏览...", command=browse).grid(row=row, column=1+col_span, sticky="w", **pad)
            row += 1
            return ent

        def pick_input():
            p = filedialog.askopenfilename(
                initialdir=self._get_initial_dir("input"),
                title="选择输入文件",
                filetypes=[
                    ("Markdown", "*.md"),
                    ("PDF", "*.pdf"),
                    ("Word", "*.docx"),
                    ("Text", "*.txt"),
                    ("All files", "*.*"),
                ],
            )
            if p:
                self.var_input.set(p)
                self._remember_path(p)
                self._remember_input_path(p)
                try:
                    self._toggle_pdf_panel()
                except Exception:
                    pass

        def pick_output():
            inp = self.var_input.get().lower()
            def_ext = ".md" if inp.endswith(".md") else ".txt"
            p = filedialog.asksaveasfilename(
                initialdir=self._get_initial_dir("output"),
                title="选择输出文件",
                defaultextension=def_ext,
                filetypes=[
                    ("Markdown", "*.md"),
                    ("Text", "*.txt"),
                    ("All files", "*.*"),
                ],
            )
            if p:
                self.var_output.set(p)
                self._remember_path(p)
                self._remember_output_path(p)

        add_entry("Base URL", self.var_base)
        add_entry("API Key", self.var_key)
        add_entry("模型名", self.var_model)
        add_entry("输入文件", self.var_input, browse=pick_input)
        add_entry("输出文件", self.var_output, browse=pick_output)
        add_entry("源语言", self.var_src, width=20, col_span=1)
        ttk.Label(frm, text="目标语言").grid(row=row-1, column=2, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_tgt, width=12).grid(row=row-1, column=3, sticky="w", **pad)

        # line: chunk size / overlap / start chunk
        ttk.Label(frm, text="分块大小").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_chunk, width=10).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frm, text="重叠字符").grid(row=row, column=2, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_overlap, width=10).grid(row=row, column=3, sticky="w", **pad)
        row += 1

        self.pdf_panel = ttk.LabelFrame(frm, text="PDF 识别选项")
        self.pdf_panel.grid(row=row, column=0, columnspan=4, sticky="nsew", **pad)
        for c in range(4):
            self.pdf_panel.columnconfigure(c, weight=1)
        ttk.Label(self.pdf_panel, text="识别方式").grid(row=0, column=0, sticky="e", **pad)
        ttk.Combobox(self.pdf_panel, textvariable=self.var_pdf_mode, values=["auto","vlm","none"], state="readonly", width=10).grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(self.pdf_panel, text="VLM URL").grid(row=0, column=2, sticky="e", **pad)
        ttk.Entry(self.pdf_panel, textvariable=self.var_vlm_url, width=32).grid(row=0, column=3, sticky="w", **pad)
        ttk.Label(self.pdf_panel, text="VLM Key").grid(row=1, column=0, sticky="e", **pad)
        ttk.Entry(self.pdf_panel, textvariable=self.var_vlm_key, show="*", width=32).grid(row=1, column=1, sticky="w", **pad)
        ttk.Label(self.pdf_panel, text="DPI").grid(row=1, column=2, sticky="e", **pad)
        ttk.Entry(self.pdf_panel, textvariable=self.var_pdf_dpi, width=10).grid(row=1, column=3, sticky="w", **pad)
        ttk.Label(self.pdf_panel, text="页范围").grid(row=2, column=0, sticky="e", **pad)
        ttk.Entry(self.pdf_panel, textvariable=self.var_pdf_pages, width=20).grid(row=2, column=1, sticky="w", **pad)
        self.pdf_panel.grid_remove()
        row += 1

        ttk.Label(frm, text="起始分块").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_start, width=10).grid(row=row, column=1, sticky="w", **pad)
        # 移除原先散落在这里的“仅将重叠当做上下文（不输出）”复选框，统一放到下方“开关选项”中
        row += 1

        ttk.Label(frm, text="超时(秒)").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_timeout, width=10).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frm, text="重试次数").grid(row=row, column=2, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_retries, width=10).grid(row=row, column=3, sticky="w", **pad)
        row += 1

        # 统一的开关与二分参数分组，确保对齐
        opts = ttk.LabelFrame(frm, text="开关选项")
        opts.grid(row=row, column=0, columnspan=4, sticky="nsew", **pad)
        # 使用 6 列布局，便于均匀分布与留白
        for c in range(12):
            opts.columnconfigure(c, weight=1, uniform="optscols")
        orow = 0
        # 第一排（4 项）：短标签，四等分
        ttk.Checkbutton(opts, text="出错后自动跳过", variable=self.var_skip).grid(row=orow, column=0, columnspan=3, sticky="w", padx=10, pady=4)
        ttk.Checkbutton(opts, text="使用流式响应", variable=self.var_stream).grid(row=orow, column=3, columnspan=3, sticky="w", padx=10, pady=4)
        ttk.Checkbutton(opts, text="隐藏思维链", variable=self.var_hide_chain).grid(row=orow, column=6, columnspan=3, sticky="w", padx=10, pady=4)
        ttk.Checkbutton(opts, text="失败自动二分", variable=self.var_bisect).grid(row=orow, column=9, columnspan=3, sticky="w", padx=10, pady=4)
        orow += 1
        # 第二排（3 项）：长标签，三等分
        ttk.Checkbutton(opts, text="追加续写 (resume)", variable=self.var_resume).grid(row=orow, column=0, columnspan=4, sticky="w", padx=10, pady=4)
        ttk.Checkbutton(opts, text="覆盖已存在输出 (overwrite)", variable=self.var_overwrite).grid(row=orow, column=4, columnspan=4, sticky="w", padx=10, pady=4)
        ttk.Checkbutton(opts, text="仅将重叠当做上下文（不输出）", variable=self.var_ctx_only).grid(row=orow, column=8, columnspan=4, sticky="w", padx=10, pady=4)
        orow += 1

        # 第三排：输入项，与第二排左侧对齐（标签占2列，输入框占2列）
        ttk.Label(opts, text="思维链标签名").grid(row=orow, column=0, columnspan=2, sticky="w", padx=(10, 8), pady=4)
        ttk.Entry(opts, textvariable=self.var_chain_tag, width=16).grid(row=orow, column=2, columnspan=2, sticky="w", padx=(0, 10), pady=4)
        ttk.Label(opts, text="二分最小长度").grid(row=orow, column=4, columnspan=2, sticky="w", padx=(10, 8), pady=4)
        ttk.Entry(opts, textvariable=self.var_minchars, width=16).grid(row=orow, column=6, columnspan=2, sticky="w", padx=(0, 10), pady=4)
        ttk.Label(opts, text="二分最大深度").grid(row=orow, column=8, columnspan=2, sticky="w", padx=(10, 8), pady=4)
        ttk.Entry(opts, textvariable=self.var_maxdepth, width=16).grid(row=orow, column=10, columnspan=2, sticky="w", padx=(0, 10), pady=4)
        orow += 1
        row += 1

        # New section: System Prompt & Hyperparameters
        lf = ttk.LabelFrame(frm, text="提示词与超参数（留空表示使用模型默认值）")
        lf.grid(row=row, column=0, columnspan=4, sticky="nsew", **pad)
        # System prompt (multi-line with scrollbar)
        ttk.Label(lf, text="系统提示词（可选）").grid(row=0, column=0, sticky="ne", padx=4, pady=4)
        sp_frame = ttk.Frame(lf)
        sp_frame.grid(row=0, column=1, columnspan=3, sticky="nsew", padx=4, pady=4)
        sp_frame.rowconfigure(0, weight=1)
        sp_frame.columnconfigure(0, weight=1)
        self.txt_sys_prompt = tk.Text(sp_frame, height=6, wrap="word")
        self.txt_sys_prompt.grid(row=0, column=0, sticky="nsew")
        sp_scroll = ttk.Scrollbar(sp_frame, orient="vertical", command=self.txt_sys_prompt.yview)
        sp_scroll.grid(row=0, column=1, sticky="ns")
        self.txt_sys_prompt.configure(yscrollcommand=sp_scroll.set)
        
        # Set default system prompt if available
        default_system_prompt = DEFAULTS.get("system_prompt", "")
        if default_system_prompt:
            self.txt_sys_prompt.insert("1.0", default_system_prompt)
        
        
        # Hyperparams inputs
        ttk.Label(lf, text="temperature").grid(row=1, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(lf, textvariable=self.var_temperature, width=10).grid(row=1, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(lf, text="top_p").grid(row=1, column=2, sticky="e", padx=4, pady=2)
        ttk.Entry(lf, textvariable=self.var_top_p, width=10).grid(row=1, column=3, sticky="w", padx=4, pady=2)

        ttk.Label(lf, text="top_k").grid(row=2, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(lf, textvariable=self.var_top_k, width=10).grid(row=2, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(lf, text="repetition_penalty").grid(row=2, column=2, sticky="e", padx=4, pady=2)
        ttk.Entry(lf, textvariable=self.var_repetition_penalty, width=10).grid(row=2, column=3, sticky="w", padx=4, pady=2)

        ttk.Label(lf, text="length_penalty").grid(row=3, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(lf, textvariable=self.var_length_penalty, width=10).grid(row=3, column=1, sticky="w", padx=4, pady=2)
        # stretch inside labelframe
        for c in range(4):
            lf.columnconfigure(c, weight=1)
        lf.rowconfigure(0, weight=1)
        row += 1

        # 控制按钮区域，右对齐，统一宽度
        ctl = ttk.Frame(frm)
        ctl.grid(row=row, column=0, columnspan=4, sticky="ew", **pad)
        
        # 使用语义化按钮样式（如果 ttkbootstrap 可用）
        start_style = "TButton"
        stop_style = "TButton"
            
        self.btn_start = ttk.Button(ctl, text="开始翻译", command=self._on_start_translate, 
                                   style=start_style, width=12)
        self.btn_start.grid(row=0, column=0, sticky="e", padx=4, pady=4)
        
        self.btn_stop = ttk.Button(ctl, text="停止", command=self._on_stop_translate, 
                                  style=stop_style, width=12)
        self.btn_stop.grid(row=0, column=1, sticky="e", padx=4, pady=4)
        
        ctl.columnconfigure(0, weight=1)  # 让按钮右对齐
        row += 1

        # Log and progress (text area with its own scrollbar)
        self.status_var = tk.StringVar(value="就绪")
        self.progress = ttk.Progressbar(frm, orient="horizontal", mode="determinate")
        self.progress.grid(row=row, column=0, columnspan=4, sticky="we", **pad)
        row += 1
        ttk.Label(frm, textvariable=self.status_var).grid(row=row, column=0, columnspan=4, sticky="w", **pad)
        row += 1

        frame_log = ttk.Frame(frm)
        frame_log.grid(row=row, column=0, columnspan=4, sticky="nsew", **pad)
        self.txt = tk.Text(frame_log, wrap="word", height=16, state=tk.DISABLED)
        vscroll = ttk.Scrollbar(frame_log, orient="vertical", command=self.txt.yview)
        self.txt.configure(yscrollcommand=vscroll.set)
        self.txt.grid(row=0, column=0, sticky="nsew")
        vscroll.grid(row=0, column=1, sticky="ns")
        # 绑定鼠标滚轮到日志区域（跨平台）
        def _wheel_txt(event, _w=self.txt):
            try:
                if getattr(event, 'num', None) == 4:
                    _w.yview_scroll(-1, "units"); return
                if getattr(event, 'num', None) == 5:
                    _w.yview_scroll(1, "units"); return
                delta = int(getattr(event, 'delta', 0))
                if delta == 0: return
                step = -1 if delta > 0 else 1
                _w.yview_scroll(step, "units")
            except Exception:
                pass
        self.txt.bind("<MouseWheel>", _wheel_txt)
        self.txt.bind("<Button-4>", _wheel_txt)
        self.txt.bind("<Button-5>", _wheel_txt)
        frame_log.rowconfigure(0, weight=1)
        frame_log.columnconfigure(0, weight=1)
        frm.rowconfigure(row, weight=1)
        for c in range(4):
            frm.columnconfigure(c, weight=1, uniform="maincols")

        # both runners: subprocess (dev) and in-process (frozen)
        self.runner = ProcessRunner(self.txt, self.progress, self.status_var)
        self.inproc = InProcessRunner(self.txt, self.progress, self.status_var)

    def _build_inspect_tab(self):
        # Use scrollable container for small screens
        sc = ScrollableFrame(self.tab_inspect)
        sc.pack(fill=tk.BOTH, expand=True)
        frm = sc.content
        pad = {"padx": 6, "pady": 4}

        self.var_ins_input = tk.StringVar(value=DEFAULTS["input"])
        self.var_ins_chunks = tk.StringVar(value="1")
        self.var_ins_maxchars = tk.IntVar(value=DEFAULTS["chunk_size"])
        self.var_ins_overlap = tk.IntVar(value=DEFAULTS["overlap_chars"])
        self.var_ins_export = tk.StringVar(value="")

        row = 0
        def pick_input():
            p = filedialog.askopenfilename(initialdir=self._get_initial_dir(), title="选择输入文件", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
            if p:
                if not p.lower().endswith(".txt"):
                    messagebox.showerror("错误", "只支持 .txt 文本文件作为输入")
                    return
                self.var_ins_input.set(p)
                self._remember_path(p)

        def pick_output():
            p = filedialog.asksaveasfilename(initialdir=self._get_initial_dir(), title="选择输出文件", defaultextension=".txt", filetypes=[("Text files", "*.txt")])
            if p:
                self.var_output.set(p)
                self._remember_path(p)
        def pick_export():
            p = filedialog.asksaveasfilename(initialdir=self._get_initial_dir(), title="导出报告为...", defaultextension=".txt")
            if p:
                self.var_ins_export.set(p)
                self._remember_path(p)

        ttk.Label(frm, text="输入文件").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_ins_input, width=64).grid(row=row, column=1, sticky="we", **pad)
        ttk.Button(frm, text="浏览...", command=pick_input).grid(row=row, column=2, sticky="w", **pad)
        row += 1

        ttk.Label(frm, text="分块序号(支持 1,2,5-8)").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_ins_chunks, width=30).grid(row=row, column=1, sticky="w", **pad)
        row += 1

        ttk.Label(frm, text="分块大小").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_ins_maxchars, width=10).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frm, text="重叠字符").grid(row=row, column=2, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_ins_overlap, width=10).grid(row=row, column=3, sticky="w", **pad)
        row += 1

        ctl = ttk.Frame(frm)
        ctl.grid(row=row, column=0, columnspan=4, sticky="we", **pad)
        ctl.columnconfigure(0, weight=1)
        btn_preview = ttk.Button(ctl, text="预览到下方日志", command=self._on_inspect_preview, width=16)
        btn_export = ttk.Button(ctl, text="导出为文件", command=self._on_inspect_export, width=12)
        btn_preview.grid(row=0, column=1, sticky="e", padx=6)
        btn_export.grid(row=0, column=2, sticky="w", padx=6)
        row += 1

        self.status_ins = tk.StringVar(value="就绪")
        ttk.Label(frm, textvariable=self.status_ins).grid(row=row, column=0, columnspan=4, sticky="w", **pad)
        row += 1

        frame_log = ttk.Frame(frm)
        frame_log.grid(row=row, column=0, columnspan=4, sticky="nsew", **pad)
        self.txt_ins = tk.Text(frame_log, wrap="word", height=18, state=tk.DISABLED)
        vscroll2 = ttk.Scrollbar(frame_log, orient="vertical", command=self.txt_ins.yview)
        self.txt_ins.configure(yscrollcommand=vscroll2.set)
        self.txt_ins.grid(row=0, column=0, sticky="nsew")
        vscroll2.grid(row=0, column=1, sticky="ns")
        # 绑定鼠标滚轮到检查页日志区域（跨平台）
        def _wheel_txt_ins(event, _w=self.txt_ins):
            try:
                if getattr(event, 'num', None) == 4:
                    _w.yview_scroll(-1, "units"); return
                if getattr(event, 'num', None) == 5:
                    _w.yview_scroll(1, "units"); return
                delta = int(getattr(event, 'delta', 0))
                if delta == 0: return
                step = -1 if delta > 0 else 1
                _w.yview_scroll(step, "units")
            except Exception:
                pass
        self.txt_ins.bind("<MouseWheel>", _wheel_txt_ins)
        self.txt_ins.bind("<Button-4>", _wheel_txt_ins)
        self.txt_ins.bind("<Button-5>", _wheel_txt_ins)
        frame_log.rowconfigure(0, weight=1)
        frame_log.columnconfigure(0, weight=1)
        frm.rowconfigure(row, weight=1)
        for c in range(4):
            frm.columnconfigure(c, weight=1)

    def _gather_translate_args(self):
        args = ["translate_lmstudio.py",
            "--input", self.var_input.get(),
            "--output", self.var_output.get(),
            "--base-url", self.var_base.get(),
            "--api-key", self.var_key.get(),
            "--model", self.var_model.get(),
            "--source-lang", self.var_src.get(),
            "--target-lang", self.var_tgt.get(),
            "--chunk-size-chars", str(self.var_chunk.get()),
            "--overlap-chars", str(self.var_overlap.get()),
            "--start-chunk", str(self.var_start.get()),
            "--request-timeout", str(self.var_timeout.get()),
            "--max-retries", str(self.var_retries.get()),
            "--bisect-min-chars", str(self.var_minchars.get()),
            "--bisect-max-depth", str(self.var_maxdepth.get()),
        ]
        # booleans as BooleanOptionalAction
        args += ["--context-only-overlap" if self.var_ctx_only.get() else "--no-context-only-overlap"]
        args += ["--skip-on-error" if self.var_skip.get() else "--no-skip-on-error"]
        args += ["--use-stream" if self.var_stream.get() else "--no-use-stream"]
        args += ["--auto-bisect" if self.var_bisect.get() else "--no-auto-bisect"]
        args += ["--overwrite" if self.var_overwrite.get() else "--no-overwrite"]
        args += ["--resume" if self.var_resume.get() else "--no-resume"]
        # 输出后处理
        args += ["--hide-chain" if self.var_hide_chain.get() else "--no-hide-chain"]
        tag = (self.var_chain_tag.get() or "think").strip()
        if tag:
            args += ["--chain-tag", tag]
        # Optional system prompt & hyperparameters
        try:
            sys_prompt = self.txt_sys_prompt.get("1.0", "end-1c").strip()
        except Exception:
            sys_prompt = ""
        if sys_prompt:
            args += ["--system-prompt", sys_prompt]
        t = self.var_temperature.get().strip()
        if t:
            args += ["--temperature", t]
        tp = self.var_top_p.get().strip()
        if tp:
            args += ["--top-p", tp]
        tkv = self.var_top_k.get().strip()
        if tkv:
            args += ["--top-k", tkv]
        rp = self.var_repetition_penalty.get().strip()
        if rp:
            args += ["--repetition-penalty", rp]
        lp = self.var_length_penalty.get().strip()
        if lp:
            args += ["--length-penalty", lp]
        inp = self.var_input.get().lower()
        if inp.endswith('.pdf'):
            args += ["--pdf-recognizer", (self.var_pdf_mode.get() or "auto")]
            if self.var_vlm_url.get():
                args += ["--vlm-url", self.var_vlm_url.get()]
            if self.var_vlm_key.get():
                args += ["--vlm-key", self.var_vlm_key.get()]
            args += ["--pdf-dpi", str(int(self.var_pdf_dpi.get()))]
            if self.var_pdf_pages.get():
                args += ["--pdf-pages", self.var_pdf_pages.get()]
        return args

    def _gather_translate_kwargs(self) -> dict:
        """将界面参数映射为 translate_lmstudio.run_translation(**kwargs) 的入参。"""
        kwargs = {
            "input_path": self.var_input.get(),
            "output_path": self.var_output.get(),
            "base_url": self.var_base.get(),
            "api_key": self.var_key.get(),
            "model": self.var_model.get(),
            "source_lang": self.var_src.get(),
            "target_lang": self.var_tgt.get(),
            "max_chars": int(self.var_chunk.get()),
            "overlap_chars": int(self.var_overlap.get()),
            "context_only_overlap": bool(self.var_ctx_only.get()),
            "start_chunk": int(self.var_start.get()),
            "request_timeout": int(self.var_timeout.get()),
            "max_retries": int(self.var_retries.get()),
            "skip_on_error": bool(self.var_skip.get()),
            "use_stream": bool(self.var_stream.get()),
            "auto_bisect_on_fail": bool(self.var_bisect.get()),
            "bisect_min_chars": int(self.var_minchars.get()),
            "bisect_max_depth": int(self.var_maxdepth.get()),
            # 输出后处理参数
            "hide_chain": bool(self.var_hide_chain.get()),
            "chain_tag": (self.var_chain_tag.get() or "think").strip(),
        }
        # Optional parameters
        try:
            sys_prompt = self.txt_sys_prompt.get("1.0", "end-1c").strip()
        except Exception:
            sys_prompt = ""
        if sys_prompt:
            kwargs["system_prompt"] = sys_prompt
        def _parse_float(s: str):
            try:
                return float(s)
            except Exception:
                return None
        def _parse_int(s: str):
            try:
                return int(float(s))
            except Exception:
                return None
        t = _parse_float(self.var_temperature.get().strip()) if self.var_temperature.get() else None
        if t is not None:
            kwargs["temperature"] = t
        tp = _parse_float(self.var_top_p.get().strip()) if self.var_top_p.get() else None
        if tp is not None:
            kwargs["top_p"] = tp
        tkv = _parse_int(self.var_top_k.get().strip()) if self.var_top_k.get() else None
        if tkv is not None:
            kwargs["top_k"] = tkv
        rp = _parse_float(self.var_repetition_penalty.get().strip()) if self.var_repetition_penalty.get() else None
        if rp is not None:
            kwargs["repetition_penalty"] = rp
        lp = _parse_float(self.var_length_penalty.get().strip()) if self.var_length_penalty.get() else None
        if lp is not None:
            kwargs["length_penalty"] = lp
        inp = self.var_input.get().lower()
        if inp.endswith('.pdf'):
            kwargs["pdf_recognition"] = {
                "mode": (self.var_pdf_mode.get() or "auto"),
                "vlm_url": self.var_vlm_url.get(),
                "vlm_key": self.var_vlm_key.get(),
                "dpi": int(self.var_pdf_dpi.get()),
                "pages": (self.var_pdf_pages.get() or None),
            }
        return kwargs

    def _on_start_translate(self):
        # Basic checks
        inp = self.var_input.get()
        if not inp or not os.path.exists(inp):
            if not messagebox.askyesno("提示", "输入文件不存在，仍然继续启动吗？"):
                return
        # 支持 .txt/.md/.pdf/.docx 输入
        if not any(inp.lower().endswith(ext) for ext in ('.txt', '.md', '.pdf', '.docx')):
            messagebox.showerror("错误", "仅支持 .txt/.md/.pdf/.docx 作为输入文件")
            return
        try:
            self._toggle_pdf_panel()
        except Exception:
            pass
        if not self.var_model.get():
            if not messagebox.askyesno("提示", "模型名未填写，仍然继续启动吗？"):
                return
        # Remember directory from input
        self._remember_path(inp)
        try:
            PREFS.setdefault("pdf", {})
            PREFS["pdf"]["mode"] = (self.var_pdf_mode.get() or "auto")
            PREFS["pdf"]["vlm_url"] = self.var_vlm_url.get()
            PREFS["pdf"]["vlm_key"] = self.var_vlm_key.get()
            PREFS["pdf"]["dpi"] = int(self.var_pdf_dpi.get())
            PREFS["pdf"]["pages"] = (self.var_pdf_pages.get() or "")
            _save_prefs(PREFS)
        except Exception:
            pass
        # Route: in-process when frozen, else subprocess
        if getattr(sys, 'frozen', False):
            kwargs = self._gather_translate_kwargs()
            self.inproc.start_translate(kwargs, overwrite=self.var_overwrite.get(), resume=self.var_resume.get())
        else:
            args = self._gather_translate_args()
            self.runner.start(args)

    def _on_stop_translate(self):
        if getattr(sys, 'frozen', False):
            self.inproc.stop()
        else:
            self.runner.stop()

    def _run_inspect(self, export_path: str | None):
        # Enforce .txt input for inspect
        path = self.var_ins_input.get()
        if not path.lower().endswith('.txt'):
            messagebox.showerror("错误", "输入文件必须是 .txt 文本文件")
            return
        if getattr(sys, 'frozen', False):
            # In-process inspect to avoid subprocess in packaged mode
            try:
                import inspect_chunks as ic
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                sentences = ic.split_sentences(text)
                pairs = ic.chunk_by_chars_pairs(sentences, int(self.var_ins_maxchars.get()), int(self.var_ins_overlap.get()))
                total = len(pairs)
                indices = ic.parse_chunk_indices(self.var_ins_chunks.get(), total)
                if not indices:
                    out = f"No valid chunk indices parsed from '{self.var_ins_chunks.get()}'. Total available: {total}\n"
                else:
                    reports: list[str] = []
                    for idx in indices:
                        ctx, content = pairs[idx - 1]
                        reports.append(ic.build_chunk_report(idx, ctx, content))
                    out = "\n".join(reports)
                if export_path:
                    with open(export_path, "w", encoding="utf-8") as out_f:
                        out_f.write(out)
                    self.status_ins.set(f"已导出到：{export_path}")
                else:
                    self.txt_ins.configure(state=tk.NORMAL)
                    self.txt_ins.delete(1.0, tk.END)
                    self.txt_ins.insert(tk.END, out)
                    self.txt_ins.configure(state=tk.DISABLED)
                    self.status_ins.set("完成")
            except Exception as e:
                self.status_ins.set("失败")
                messagebox.showerror("错误", str(e))
            return

        # Fallback: subprocess mode for dev
        args = ["inspect_chunks.py", "--input", path, "--chunks", self.var_ins_chunks.get(),
                "--max-chars", str(self.var_ins_maxchars.get()), "--overlap", str(self.var_ins_overlap.get())]
        if export_path:
            args += ["--export", export_path]
        # Choose output text widget based on export
        if export_path:
            # fire and wait; then show status
            self.status_ins.set("运行中...")
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            try:
                cp = subprocess.run([PYTHON_EXE] + args, cwd=str(SCRIPT_DIR), text=True, encoding="utf-8", errors="replace", capture_output=True, env=env)
                if cp.returncode == 0:
                    self.status_ins.set(f"已导出到：{export_path}")
                else:
                    self.status_ins.set(f"失败（退出码 {cp.returncode}）")
                    messagebox.showerror("错误", cp.stderr or "inspect 失败")
            except Exception as e:
                self.status_ins.set("失败")
                messagebox.showerror("错误", str(e))
        else:
            # capture and print to txt_ins
            self.status_ins.set("运行中...")
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            try:
                cp = subprocess.run([PYTHON_EXE] + args, cwd=str(SCRIPT_DIR), text=True, encoding="utf-8", errors="replace", capture_output=True, env=env)
                self.txt_ins.configure(state=tk.NORMAL)
                self.txt_ins.delete(1.0, tk.END)
                out = cp.stdout or ""
                err = cp.stderr or ""
                self.txt_ins.insert(tk.END, out)
                if err:
                    self.txt_ins.insert(tk.END, "\n[ERR] " + err)
                self.txt_ins.configure(state=tk.DISABLED)
                if cp.returncode == 0:
                    self.status_ins.set("完成")
                else:
                    self.status_ins.set(f"完成（退出码 {cp.returncode}）")
            except Exception as e:
                self.status_ins.set("失败")
                messagebox.showerror("错误", str(e))

    def _on_inspect_preview(self):
        self._run_inspect(export_path=None)

    def _on_inspect_export(self):
        p = self.var_ins_export.get()
        if not p:
            p = filedialog.asksaveasfilename(initialdir=self._get_initial_dir(), title="导出报告为...", defaultextension=".txt")
            if not p:
                return
        self._run_inspect(export_path=p)


    def _on_theme_change(self, event=None):
        """主题切换回调函数"""
        try:
            import ttkbootstrap as tb
            new_theme = self.var_theme.get()
            # 优先复用已持有的 Style；若不存在则创建一次
            if getattr(self, "_tb_style", None) is None:
                self._tb_style = tb.Style(theme=new_theme)
            else:
                try:
                    self._tb_style.theme_use(new_theme)
                except Exception:
                    # 某些环境下 theme_use 不可用，退回重新创建
                    self._tb_style = tb.Style(theme=new_theme)
            self._apply_custom_styles(self._tb_style)
            # 保存主题偏好
            PREFS["theme"] = new_theme
            _save_prefs(PREFS)
        except Exception as e:
            messagebox.showerror("主题切换失败", f"无法切换到主题 '{new_theme}':\n{e}")

    def _toggle_pdf_panel(self):
        p = self.var_input.get().lower()
        if p.endswith('.pdf'):
            self.pdf_panel.grid()
        else:
            self.pdf_panel.grid_remove()
    
    def _setup_text_hover_effects(self, text_widget):
        """为Text控件设置鼠标悬停效果"""
        def on_enter(event):
            text_widget.configure(bg="#f0f8ff")  # 浅蓝色背景
        
        def on_leave(event):
            text_widget.configure(bg="white")  # 恢复白色背景
        
        def on_focus_in(event):
            text_widget.configure(bg="#e6f3ff")  # 聚焦时稍深的蓝色
        
        def on_focus_out(event):
            text_widget.configure(bg="white")  # 失焦时恢复白色
        
        # 绑定事件
        text_widget.bind("<Enter>", on_enter)
        text_widget.bind("<Leave>", on_leave)
        text_widget.bind("<FocusIn>", on_focus_in)
        text_widget.bind("<FocusOut>", on_focus_out)
    
    def _apply_custom_styles(self, style):
        """应用自定义样式配置"""
        try:
            style.configure("TNotebook.Tab", padding=(12, 6), font=("Segoe UI", 9))
            style.configure("TButton", padding=(10, 6), font=("Segoe UI", 9))
            style.configure("TCheckbutton", padding=(4, 3), font=("Segoe UI", 9))
            style.configure("TLabelframe", padding=(12, 8), relief="flat", borderwidth=1)
            style.configure("TLabelframe.Label", padding=(6, 2), font=("Segoe UI", 9, "bold"))
            style.configure("TLabel", font=("Segoe UI", 9))
            style.configure("TEntry", padding=(6, 4), font=("Segoe UI", 9))
            
            # 语义化按钮样式
            try:
                style.configure("success.TButton", font=("Segoe UI", 9, "bold"))
                style.configure("danger.TButton", font=("Segoe UI", 9, "bold"))
                style.configure("info.TButton", font=("Segoe UI", 9))
            except:
                pass
        except Exception:
            pass

if __name__ == "__main__":
    app = App()
    app.mainloop()
