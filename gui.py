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

# Use executable directory when packaged (PyInstaller), else file directory
if getattr(sys, 'frozen', False):
    SCRIPT_DIR = Path(sys.executable).resolve().parent
else:
    SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PY = (SCRIPT_DIR / ".venv" / "Scripts" / "python.exe")
PYTHON_EXE = str(DEFAULT_PY if DEFAULT_PY.exists() else sys.executable)

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
    })
except SystemExit:
    # openai missing during import; fall back to built-in defaults
    pass
except Exception:
    pass

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

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LMStudio 翻译助手 GUI")
        self.geometry("980x680")
        self.minsize(900, 600)

        self.notebook = ttk.Notebook(self)
        self.tab_translate = ttk.Frame(self.notebook)
        self.tab_inspect = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_translate, text="开始翻译")
        self.notebook.add(self.tab_inspect, text="检查分块")
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self._build_translate_tab()
        self._build_inspect_tab()

    def _build_translate_tab(self):
        frm = self.tab_translate
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

        # Grid: 3 columns labels/entries/buttons
        row = 0
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
            p = filedialog.askopenfilename(initialdir=str(SCRIPT_DIR), title="选择输入文件", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
            if p:
                if not p.lower().endswith(".txt"):
                    messagebox.showerror("错误", "只支持 .txt 文本文件作为输入")
                    return
                self.var_input.set(p)

        def pick_output():
            p = filedialog.asksaveasfilename(initialdir=str(SCRIPT_DIR), title="选择输出文件", defaultextension=".txt", filetypes=[("Text files", "*.txt")])
            if p:
                self.var_output.set(p)

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

        ttk.Label(frm, text="起始分块").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_start, width=10).grid(row=row, column=1, sticky="w", **pad)
        ttk.Checkbutton(frm, text="仅将重叠当做上下文（不输出）", variable=self.var_ctx_only).grid(row=row, column=2, columnspan=2, sticky="w", **pad)
        row += 1

        ttk.Label(frm, text="超时(秒)").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_timeout, width=10).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frm, text="重试次数").grid(row=row, column=2, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_retries, width=10).grid(row=row, column=3, sticky="w", **pad)
        row += 1

        ttk.Checkbutton(frm, text="出错后自动跳过", variable=self.var_skip).grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        ttk.Checkbutton(frm, text="使用流式响应", variable=self.var_stream).grid(row=row, column=2, columnspan=2, sticky="w", **pad)
        row += 1

        ttk.Checkbutton(frm, text="失败自动二分", variable=self.var_bisect).grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        ttk.Label(frm, text="二分最小长度").grid(row=row, column=2, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_minchars, width=10).grid(row=row, column=3, sticky="w", **pad)
        row += 1

        ttk.Label(frm, text="二分最大深度").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_maxdepth, width=10).grid(row=row, column=1, sticky="w", **pad)
        ttk.Checkbutton(frm, text="覆盖已存在输出 (overwrite)", variable=self.var_overwrite).grid(row=row, column=2, sticky="w", **pad)
        ttk.Checkbutton(frm, text="追加续写 (resume)", variable=self.var_resume).grid(row=row, column=3, sticky="w", **pad)
        row += 1

        # Controls: Start / Stop
        ctl = ttk.Frame(frm)
        ctl.grid(row=row, column=0, columnspan=4, sticky="we", **pad)
        self.btn_start = ttk.Button(ctl, text="开始翻译", command=self._on_start_translate)
        self.btn_stop = ttk.Button(ctl, text="停止", command=self._on_stop_translate)
        self.btn_start.pack(side=tk.LEFT, padx=6)
        self.btn_stop.pack(side=tk.LEFT, padx=6)
        row += 1

        # Log and progress
        self.status_var = tk.StringVar(value="就绪")
        self.progress = ttk.Progressbar(frm, orient="horizontal", mode="determinate")
        self.progress.grid(row=row, column=0, columnspan=4, sticky="we", **pad)
        row += 1
        ttk.Label(frm, textvariable=self.status_var).grid(row=row, column=0, columnspan=4, sticky="w", **pad)
        row += 1

        self.txt = tk.Text(frm, wrap="word", height=18, state=tk.DISABLED)
        self.txt.grid(row=row, column=0, columnspan=4, sticky="nsew", **pad)
        frm.rowconfigure(row, weight=1)
        for c in range(4):
            frm.columnconfigure(c, weight=1)

        # both runners: subprocess (dev) and in-process (frozen)
        self.runner = ProcessRunner(self.txt, self.progress, self.status_var)
        self.inproc = InProcessRunner(self.txt, self.progress, self.status_var)

    def _build_inspect_tab(self):
        frm = self.tab_inspect
        pad = {"padx": 6, "pady": 4}

        self.var_ins_input = tk.StringVar(value=DEFAULTS["input"])
        self.var_ins_chunks = tk.StringVar(value="1")
        self.var_ins_maxchars = tk.IntVar(value=DEFAULTS["chunk_size"])
        self.var_ins_overlap = tk.IntVar(value=DEFAULTS["overlap_chars"])
        self.var_ins_export = tk.StringVar(value="")

        row = 0
        def pick_input():
            p = filedialog.askopenfilename(initialdir=str(SCRIPT_DIR), title="选择输入文件", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
            if p:
                if not p.lower().endswith(".txt"):
                    messagebox.showerror("错误", "只支持 .txt 文本文件作为输入")
                    return
                self.var_ins_input.set(p)

        def pick_output():
            p = filedialog.asksaveasfilename(initialdir=str(SCRIPT_DIR), title="选择输出文件", defaultextension=".txt", filetypes=[("Text files", "*.txt")])
            if p:
                self.var_output.set(p)
        def pick_export():
            p = filedialog.asksaveasfilename(initialdir=str(SCRIPT_DIR), title="导出报告为...", defaultextension=".txt")
            if p:
                self.var_ins_export.set(p)

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
        ttk.Button(ctl, text="预览到下方日志", command=self._on_inspect_preview).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctl, text="导出为文件", command=self._on_inspect_export).pack(side=tk.LEFT, padx=6)
        row += 1

        self.status_ins = tk.StringVar(value="就绪")
        ttk.Label(frm, textvariable=self.status_ins).grid(row=row, column=0, columnspan=4, sticky="w", **pad)
        row += 1

        self.txt_ins = tk.Text(frm, wrap="word", height=18, state=tk.DISABLED)
        self.txt_ins.grid(row=row, column=0, columnspan=4, sticky="nsew", **pad)
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
        return args

    def _gather_translate_kwargs(self) -> dict:
        """将界面参数映射为 translate_lmstudio.run_translation(**kwargs) 的入参。"""
        return {
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
        }

    def _on_start_translate(self):
        # Basic checks
        inp = self.var_input.get()
        if not inp or not os.path.exists(inp):
            if not messagebox.askyesno("提示", "输入文件不存在，仍然继续启动吗？"):
                return
        # Enforce .txt input
        if not inp.lower().endswith('.txt'):
            messagebox.showerror("错误", "输入文件必须是 .txt 文本文件")
            return
        if not self.var_model.get():
            if not messagebox.askyesno("提示", "模型名未填写，仍然继续启动吗？"):
                return
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
            p = filedialog.asksaveasfilename(initialdir=str(SCRIPT_DIR), title="导出报告为...", defaultextension=".txt")
            if not p:
                return
        self._run_inspect(export_path=p)


if __name__ == "__main__":
    app = App()
    app.mainloop()